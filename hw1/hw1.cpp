#include <algorithm>
#include <cctype>
#include <climits>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>
#include <mutex>
#include <atomic>
#include <set>
#include <thread>
#include <chrono>
#include <oneapi/tbb/concurrent_priority_queue.h>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <boost/functional/hash.hpp>

using namespace std;
namespace tbb = oneapi::tbb;

struct State {
    vector<string> board;
    int player_y = 0;
    int player_x = 0;

    bool operator==(const State &other) const {
        return player_y == other.player_y && player_x == other.player_x && board == other.board;
    }
};

struct StateHash {
    size_t operator()(const State &state) const {
        size_t seed = 0;
        boost::hash_combine(seed, state.player_y);
        boost::hash_combine(seed, state.player_x);
        boost::hash_combine(seed, state.board);
        return seed;
    }
};

// Compact State: Memory-efficient representation
// Stores only box positions and player position as encoded integers
struct CompactState {
    vector<uint16_t> boxes;  // Sorted list of box positions (encoded as y*COLS+x)
    uint16_t player_pos;     // Player position (encoded as y*COLS+x)
    
    CompactState() : player_pos(0) {}
    
    bool operator==(const CompactState &other) const {
        return player_pos == other.player_pos && boxes == other.boxes;
    }
};

struct CompactStateHash {
    size_t operator()(const CompactState &state) const {
        size_t seed = 0;
        boost::hash_combine(seed, state.player_pos);
        for (uint16_t box : state.boxes) {
            boost::hash_combine(seed, box);
        }
        return seed;
    }
};

namespace {
vector<string> baseBoard;
vector<vector<bool>> targetMap;
vector<vector<bool>> deadCellMap;
vector<pair<int, int>> targetPositions;
int ROWS = 0;
int COLS = 0;
bool hasFragileTile = false;

// Helper functions for position encoding/decoding
inline uint16_t encodePos(int y, int x) {
    return static_cast<uint16_t>(y * COLS + x);
}

inline pair<int, int> decodePos(uint16_t pos) {
    return {pos / COLS, pos % COLS};
}

inline bool isWithin(int y, int x) {
    return y >= 0 && y < ROWS && x >= 0 && x < COLS;
}

inline bool isWallForBox(int y, int x) {
    if (!isWithin(y, x)) {
        return true;
    }
    char tile = baseBoard[y][x];
    return tile == '#' || tile == '@';
}

// Improved Simple Deadlock Detection (conservative approach)
// Combines corner/corridor detection with limited reverse reachability
void computeSimpleDeadlocks() {
    deadCellMap.assign(ROWS, vector<bool>(COLS, false));
    if (ROWS == 0 || COLS == 0) {
        return;
    }

    // Step 1: Mark corner deadlocks (always safe)
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (baseBoard[y][x] == '#' || targetMap[y][x]) {
                continue;
            }
            bool up = isWallForBox(y - 1, x);
            bool down = isWallForBox(y + 1, x);
            bool left = isWallForBox(y, x - 1);
            bool right = isWallForBox(y, x + 1);
            // Corner deadlocks
            if ((up && left) || (up && right) || (down && left) || (down && right)) {
                deadCellMap[y][x] = true;
            }
        }
    }

    // Step 2: Mark corridor deadlocks (horizontal)
    for (int y = 0; y < ROWS; ++y) {
        int x = 0;
        while (x < COLS) {
            if (baseBoard[y][x] == '#') {
                ++x;
                continue;
            }
            vector<int> cells;
            bool corridor = true;
            bool hasTarget = false;
            while (x < COLS && baseBoard[y][x] != '#') {
                bool up = isWallForBox(y - 1, x);
                bool down = isWallForBox(y + 1, x);
                if (!(up && down)) {
                    corridor = false;
                }
                if (targetMap[y][x]) {
                    hasTarget = true;
                }
                cells.push_back(x);
                ++x;
            }
            if (corridor && !hasTarget) {
                for (int cx : cells) {
                    if (!targetMap[y][cx]) {
                        deadCellMap[y][cx] = true;
                    }
                }
            }
        }
    }

    // Step 3: Mark corridor deadlocks (vertical)
    for (int x = 0; x < COLS; ++x) {
        int y = 0;
        while (y < ROWS) {
            if (baseBoard[y][x] == '#') {
                ++y;
                continue;
            }
            vector<int> cells;
            bool corridor = true;
            bool hasTarget = false;
            while (y < ROWS && baseBoard[y][x] != '#') {
                bool left = isWallForBox(y, x - 1);
                bool right = isWallForBox(y, x + 1);
                if (!(left && right)) {
                    corridor = false;
                }
                if (targetMap[y][x]) {
                    hasTarget = true;
                }
                cells.push_back(y);
                ++y;
            }
            if (corridor && !hasTarget) {
                for (int cy : cells) {
                    if (!targetMap[cy][x]) {
                        deadCellMap[cy][x] = true;
                    }
                }
            }
        }
    }
}
} // namespace

// Convert full State to CompactState
CompactState compressState(const State &state) {
    CompactState compact;
    compact.player_pos = encodePos(state.player_y, state.player_x);
    
    // Collect all box positions
    for (int y = 0; y < static_cast<int>(state.board.size()); ++y) {
        for (int x = 0; x < static_cast<int>(state.board[y].size()); ++x) {
            char c = state.board[y][x];
            if (c == 'x' || c == 'X') {
                compact.boxes.push_back(encodePos(y, x));
            }
        }
    }
    
    // Sort boxes for consistent representation
    sort(compact.boxes.begin(), compact.boxes.end());
    
    return compact;
}

// Convert CompactState to full State (for checking win condition, output, etc.)
State decompressState(const CompactState &compact) {
    State state;
    state.board = baseBoard;  // Start from base board
    
    // Place boxes
    for (uint16_t boxPos : compact.boxes) {
        auto [y, x] = decodePos(boxPos);
        if (targetMap[y][x]) {
            state.board[y][x] = 'X';  // Box on target
        } else {
            state.board[y][x] = 'x';
        }
    }
    
    // Place player
    auto [py, px] = decodePos(compact.player_pos);
    state.player_y = py;
    state.player_x = px;
    char &cell = state.board[py][px];
    if (cell == '.') {
        cell = 'O';
    } else if (cell == 'X') {
        cell = 'O';  // Player on box on target (shouldn't happen in valid states)
    } else if (cell == 'x') {
        cell = 'o';  // Player on box (shouldn't happen)
    } else {
        cell = 'o';
    }
    
    return state;
}

// Check if compact state is solved (all boxes on targets)
bool isSolvedCompact(const CompactState &compact) {
    for (uint16_t boxPos : compact.boxes) {
        auto [y, x] = decodePos(boxPos);
        if (!targetMap[y][x]) {
            return false;  // Found a box not on target
        }
    }
    return true;
}

State loadState(const string &filename) {
    ifstream file(filename);
    State state;
    baseBoard.clear();
    targetMap.clear();
    deadCellMap.clear();
    hasFragileTile = false;

    string line;
    int y = 0;
    while (getline(file, line)) {
        state.board.push_back(line);
        string staticLine = line;
        vector<bool> targetRow;
        for (int x = 0; x < static_cast<int>(staticLine.size()); ++x) {
            char ch = line[x];
            char baseChar = ch;
            switch (ch) {
            case 'x':
            case 'o':
                baseChar = ' ';
                break;
            case 'X':
            case 'O':
                baseChar = '.';
                break;
            case '!':
                baseChar = '@';
                hasFragileTile = true;
                break;
            case '@':
                hasFragileTile = true;
                break;
            default:
                break;
            }
            staticLine[x] = baseChar;
            targetRow.push_back(ch == '.' || ch == 'O' || ch == 'X');
            unsigned char lowered = static_cast<unsigned char>(ch);
            if (ch == '!') {
                state.player_y = y;
                state.player_x = x;
            } else if (std::tolower(lowered) == 'o') {
                state.player_y = y;
                state.player_x = x;
            }
        }
        baseBoard.push_back(staticLine);
        targetMap.push_back(targetRow);
        ++y;
    }

    ROWS = static_cast<int>(baseBoard.size());
    COLS = ROWS > 0 ? static_cast<int>(baseBoard[0].size()) : 0;
    
    // Precompute target positions for heuristic (must be done before deadlock detection)
    targetPositions.clear();
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (targetMap[y][x]) {
                targetPositions.push_back({y, x});
            }
        }
    }
    
    // Compute simple deadlocks using reverse BFS from goals
    computeSimpleDeadlocks();

    return state;
}

bool isSolved(const State &state) {
    for (const auto &row : state.board) {
        for (char c : row) {
            if (c == 'x') {
                return false;
            }
        }
    }
    return true;
}

// Check if position is a wall (NOT box)
inline bool isWall(int y, int x) {
    if (!isWithin(y, x)) return true;
    return baseBoard[y][x] == '#' || baseBoard[y][x] == '@';
}

// Helper: Check if there's a box at position
inline bool isBox(const State &state, int y, int x) {
    if (!isWithin(y, x)) return false;
    char c = state.board[y][x];
    return c == 'x' || c == 'X';
}

// Enhanced Freeze Deadlock Detection with recursive box checking
// Checks if a box is blocked along a specific axis (horizontal or vertical)
// visited: tracks boxes being checked to prevent infinite recursion
bool isBlockedAlongAxis(const State &state, int y, int x, bool checkHorizontal, 
                        set<pair<int,int>> &visited) {
    // Already checked this box in current recursion path
    if (visited.count({y, x})) {
        return true;  // Treat as wall to avoid circular check
    }
    visited.insert({y, x});
    
    if (checkHorizontal) {
        // Check horizontal axis (left and right)
        bool leftBlocked = false;
        bool rightBlocked = false;
        
        // Left side
        if (isWall(y, x-1)) {
            leftBlocked = true;
        } else if (isWithin(y, x-1) && deadCellMap[y][x-1]) {
            leftBlocked = true;  // Simple deadlock square
        } else if (isBox(state, y, x-1)) {
            // Check if the box on the left is also blocked (switch to vertical axis)
            leftBlocked = isBlockedAlongAxis(state, y, x-1, false, visited);
        }
        
        // Right side
        if (isWall(y, x+1)) {
            rightBlocked = true;
        } else if (isWithin(y, x+1) && deadCellMap[y][x+1]) {
            rightBlocked = true;  // Simple deadlock square
        } else if (isBox(state, y, x+1)) {
            // Check if the box on the right is also blocked (switch to vertical axis)
            rightBlocked = isBlockedAlongAxis(state, y, x+1, false, visited);
        }
        
        return leftBlocked && rightBlocked;
        
    } else {
        // Check vertical axis (up and down)
        bool upBlocked = false;
        bool downBlocked = false;
        
        // Up side
        if (isWall(y-1, x)) {
            upBlocked = true;
        } else if (isWithin(y-1, x) && deadCellMap[y-1][x]) {
            upBlocked = true;  // Simple deadlock square
        } else if (isBox(state, y-1, x)) {
            // Check if the box above is also blocked (switch to horizontal axis)
            upBlocked = isBlockedAlongAxis(state, y-1, x, true, visited);
        }
        
        // Down side
        if (isWall(y+1, x)) {
            downBlocked = true;
        } else if (isWithin(y+1, x) && deadCellMap[y+1][x]) {
            downBlocked = true;  // Simple deadlock square
        } else if (isBox(state, y+1, x)) {
            // Check if the box below is also blocked (switch to horizontal axis)
            downBlocked = isBlockedAlongAxis(state, y+1, x, true, visited);
        }
        
        return upBlocked && downBlocked;
    }
}

// Check if box is frozen (enhanced version with recursive box checking)
bool isFrozenBox(const State &state, int y, int x) {
    if (state.board[y][x] != 'x') return false;
    if (targetMap[y][x]) return false; // Already on target, not a problem
    
    // First do a quick check with simple rules (walls only)
    bool simpleH = (isWall(y, x-1) && isWall(y, x+1));
    bool simpleV = (isWall(y-1, x) && isWall(y+1, x));
    
    // If frozen by walls only, definitely deadlock
    if (simpleH && simpleV) {
        return true;
    }
    
    // For more complex cases, limit recursion depth to avoid excessive computation
    // Only do enhanced check if there are boxes nearby
    bool hasNearbyBox = isBox(state, y-1, x) || isBox(state, y+1, x) || 
                        isBox(state, y, x-1) || isBox(state, y, x+1);
    
    if (!hasNearbyBox) {
        return false;  // No nearby boxes, can't be frozen by boxes
    }
    
    // Thread-safe: each call has its own visited set
    set<pair<int,int>> visited;
    
    // Check if frozen along both axes (with recursion limit via visited set)
    bool frozenH = isBlockedAlongAxis(state, y, x, true, visited);
    
    // Optimization: only check vertical if horizontal is already blocked
    if (!frozenH) return false;
    
    visited.clear();  // Clear for vertical check
    bool frozenV = isBlockedAlongAxis(state, y, x, false, visited);
    
    // Only deadlock if frozen in BOTH horizontal AND vertical
    return frozenH && frozenV;
}

// Check if current state is a deadlock using precomputed simple deadlocks
bool isDeadState(const State &state) {
    for (int y = 0; y < static_cast<int>(state.board.size()); ++y) {
        for (int x = 0; x < static_cast<int>(state.board[y].size()); ++x) {
            char c = state.board[y][x];
            if (c == 'x' || c == 'X') {
                // Check if box is in a simple deadlock square (precomputed)
                if (y < static_cast<int>(deadCellMap.size()) && x < static_cast<int>(deadCellMap[y].size()) &&
                    deadCellMap[y][x]) {
                    return true;
                }
                // Check if box is on a fragile tile (@)
                if (baseBoard[y][x] == '@') {
                    return true;
                }
                // Advanced deadlock: frozen box (only for boxes not on targets)
                if (c == 'x' && isFrozenBox(state, y, x)) {
                    return true;
                }
            }
        }
    }
    return false;
}

const int dy[] = {-1, 0, 1, 0};
const int dx[] = {0, -1, 0, 1};
const char directions[] = {'W', 'A', 'S', 'D'};

int dirIndexFromChar(char c) {
    switch (c) {
    case 'W':
        return 0;
    case 'A':
        return 1;
    case 'S':
        return 2;
    case 'D':
        return 3;
    default:
        return -1;
    }
}

bool tryMove(const State &current, int dir, State &out) {
    int ny = current.player_y + dy[dir];
    int nx = current.player_x + dx[dir];
    int nny = ny + dy[dir];
    int nnx = nx + dx[dir];

    if (ny < 0 || ny >= static_cast<int>(current.board.size()) || nx < 0 || nx >= static_cast<int>(current.board[0].size())) {
        return false;
    }

    out = current;
    char target = current.board[ny][nx];

    if (target == ' ') {
        out.board[ny][nx] = 'o';
    } else if (target == '.') {
        out.board[ny][nx] = 'O';
    } else if (target == '@') {
        out.board[ny][nx] = '!';
    } else if (target == 'x' || target == 'X') {
        if (nny < 0 || nny >= static_cast<int>(current.board.size()) || nnx < 0 || nnx >= static_cast<int>(current.board[0].size())) {
            return false;
        }
        char boxTarget = current.board[nny][nnx];
        if (boxTarget != ' ' && boxTarget != '.') {
            return false;
        }
        if (target == 'x') {
            out.board[ny][nx] = 'o';
        } else {
            out.board[ny][nx] = 'O';
        }
        if (boxTarget == ' ') {
            out.board[nny][nnx] = 'x';
        } else {
            out.board[nny][nnx] = 'X';
        }
    } else {
        return false;
    }

    char oldPos = current.board[current.player_y][current.player_x];
    if (oldPos == 'o') {
        out.board[current.player_y][current.player_x] = ' ';
    } else if (oldPos == '!') {
        out.board[current.player_y][current.player_x] = '@';
    } else {
        out.board[current.player_y][current.player_x] = '.';
    }

    out.player_y = ny;
    out.player_x = nx;
    return true;
}

struct ReachableInfo {
    vector<int> parent;
    vector<char> move;
    int startIndex = -1;
};

ReachableInfo computeReachable(const State &state) {
    ReachableInfo info;
    int total = ROWS * COLS;
    info.parent.assign(total, -1);
    info.move.assign(total, '?');

    if (total == 0) {
        return info;
    }

    queue<int> q;
    int startIdx = state.player_y * COLS + state.player_x;
    info.parent[startIdx] = startIdx;
    info.startIndex = startIdx;
    q.push(startIdx);

    while (!q.empty()) {
        int idx = q.front();
        q.pop();
        int y = idx / COLS;
        int x = idx % COLS;

        for (int dir = 0; dir < 4; ++dir) {
            int ny = y + dy[dir];
            int nx = x + dx[dir];
            if (!isWithin(ny, nx)) {
                continue;
            }
            char tile = state.board[ny][nx];
            if (tile == '#' || tile == 'x' || tile == 'X') {
                continue;
            }
            int nidx = ny * COLS + nx;
            if (info.parent[nidx] != -1) {
                continue;
            }
            info.parent[nidx] = idx;
            info.move[nidx] = directions[dir];
            q.push(nidx);
        }
    }

    return info;
}

string reconstructPath(int idx, const ReachableInfo &info) {
    string path;
    if (idx == -1 || info.parent[idx] == -1) {
        return path;
    }
    while (idx != info.startIndex) {
        path.push_back(info.move[idx]);
        idx = info.parent[idx];
    }
    reverse(path.begin(), path.end());
    return path;
}

bool applyMoves(const State &start, const string &moves, State &out) {
    State current = start;
    for (char mv : moves) {
        int dir = dirIndexFromChar(mv);
        if (dir < 0) {
            return false;
        }
        State next;
        if (!tryMove(current, dir, next)) {
            return false;
        }
        current = next;
    }
    out = current;
    return true;
}

// Hungarian Algorithm (Kuhn-Munkres) for minimum cost bipartite matching, O(n^3)
int hungarian(const vector<vector<int>> &cost) {
    if (cost.empty() || cost[0].empty()) return 0;
    
    int n = static_cast<int>(cost.size());
    int m = static_cast<int>(cost[0].size());
    int N = max(n, m); // Expand to square matrix
    
    vector<vector<int>> a(N, vector<int>(N, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] = cost[i][j];
        }
    }

    const int INF = 1e9;
    vector<int> u(N+1), v(N+1), p(N+1), way(N+1);
    
    for (int i = 1; i <= N; i++) {
        p[0] = i;
        vector<int> minv(N+1, INF);
        vector<char> used(N+1, false);
        int j0 = 0;
        
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INF, j1 = 0;
            
            for (int j = 1; j <= N; j++) {
                if (!used[j]) {
                    int cur = a[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            
            for (int j = 0; j <= N; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    
    return -v[0]; // Return minimum matching cost
}

// A* heuristic for CompactState with adaptive matching
int calculateHeuristicCompact(const CompactState &compact) {
    int n = static_cast<int>(compact.boxes.size());
    int m = static_cast<int>(targetPositions.size());
    
    if (n == 0) return 0;
    
    int matchCost = 0;
    
    // Use Hungarian algorithm only for complex cases (many boxes)
    if (n >= 5 && n <= 15) {
        // Build cost matrix (box → target Manhattan distance)
        vector<vector<int>> cost(n, vector<int>(m, 0));
        for (int i = 0; i < n; ++i) {
            auto [by, bx] = decodePos(compact.boxes[i]);
            for (int j = 0; j < m; ++j) {
                auto [ty, tx] = targetPositions[j];
                cost[i][j] = abs(by - ty) + abs(bx - tx);
            }
        }
        matchCost = hungarian(cost);
    } else {
        // For simple cases, use greedy matching (faster)
        for (uint16_t boxPos : compact.boxes) {
            auto [by, bx] = decodePos(boxPos);
            int minDist = INT_MAX;
            for (const auto &[ty, tx] : targetPositions) {
                int dist = abs(by - ty) + abs(bx - tx);
                minDist = min(minDist, dist);
            }
            matchCost += minDist;
        }
    }
    
    // Add player-to-nearest-box distance (divided by 2 to remain admissible)
    auto [py, px] = decodePos(compact.player_pos);
    int minPlayerDist = INT_MAX;
    for (uint16_t boxPos : compact.boxes) {
        auto [by, bx] = decodePos(boxPos);
        int dist = abs(py - by) + abs(px - bx);
        minPlayerDist = min(minPlayerDist, dist);
    }
    
    if (minPlayerDist == INT_MAX) minPlayerDist = 0;
    
    return matchCost + (minPlayerDist / 2);
}

// A* heuristic with adaptive matching: Hungarian for complex cases, greedy for simple ones
// Balances accuracy and performance (Legacy wrapper for State)
int calculateHeuristic(const State &state) {
    return calculateHeuristicCompact(compressState(state));
}

// Priority queue item for A*
struct PQItem {
    int stateIdx;
    int priority; // f = g + h
    
    bool operator>(const PQItem &other) const {
        return priority > other.priority;
    }
};

string buildSolutionPath(int idx, const vector<int> &parents, const vector<char> &pushDirs, const vector<int> &prePushIndices,
                         const vector<State> &states) {
    vector<string> segments;
    while (idx != -1 && parents[idx] != -1) {
        int parentIdx = parents[idx];
        string segment;
        int target = prePushIndices[idx];
        if (target != -1) {
            ReachableInfo reach = computeReachable(states[parentIdx]);
            segment = reconstructPath(target, reach);
        }
        if (pushDirs[idx] != '?') {
            segment.push_back(pushDirs[idx]);
        }
        segments.push_back(std::move(segment));
        idx = parentIdx;
    }
    string result;
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
        result += *it;
    }
    return result;
}

// Build solution path from CompactState vector
string buildSolutionPathCompact(int idx, const vector<int> &parents, const vector<char> &pushDirs, const vector<int> &prePushIndices,
                         const vector<CompactState> &states) {
    vector<string> segments;
    while (idx != -1 && parents[idx] != -1) {
        int parentIdx = parents[idx];
        string segment;
        int target = prePushIndices[idx];
        if (target != -1) {
            State parentState = decompressState(states[parentIdx]);
            ReachableInfo reach = computeReachable(parentState);
            segment = reconstructPath(target, reach);
        }
        if (pushDirs[idx] != '?') {
            segment.push_back(pushDirs[idx]);
        }
        segments.push_back(std::move(segment));
        idx = parentIdx;
    }
    string result;
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
        result += *it;
    }
    return result;
}

// Candidate move structure for parallel processing
struct CandidateMove {
    State afterWalk;
    int idx;
    int dir;
    int by, bx, ty, tx;
};

// TBB Concurrent BFS: Fully parallelized A* with CompactState and thread-safe containers
string solveWithConcurrentBFS(const State &initialState, bool enableDeadCheck) {
    if (enableDeadCheck && isDeadState(initialState)) {
        return "";
    }

    // TBB concurrent containers - thread-safe without explicit locks!
    tbb::concurrent_priority_queue<PQItem, greater<PQItem>> pq;
    vector<CompactState> states;  // Use CompactState for memory efficiency
    vector<int> parents;
    vector<char> pushDirs;
    vector<int> prePushIndices;
    vector<int> gScore;
    
    // Mutex only for vector operations (vectors are not thread-safe)
    mutex states_mtx;
    
    // TBB concurrent map - no lock needed for most operations!
    tbb::concurrent_unordered_map<CompactState, int, CompactStateHash> visited;
    
    // Atomic flags for synchronization
    atomic<bool> solution_found(false);
    atomic<int> solution_idx(-1);
    atomic<int> active_threads(0);
    
    // Initialize with CompactState
    CompactState initialCompact = compressState(initialState);
    states.push_back(initialCompact);
    parents.push_back(-1);
    pushDirs.emplace_back('?');
    prePushIndices.emplace_back(-1);
    gScore.push_back(0);
    
    visited.insert(make_pair(initialCompact, 0));
    
    int h0 = calculateHeuristicCompact(initialCompact);
    pq.push({0, h0});
    
    // Worker function for each thread with batch processing for better load balancing
    auto worker = [&]() {
        active_threads++;
        
        while (!solution_found.load()) {
            // Try to get a batch of items for better parallelization
            vector<PQItem> batch;
            const int batch_size = 4;  // Process multiple items per iteration
            
            for (int i = 0; i < batch_size && !solution_found.load(); ++i) {
            PQItem item;
                if (pq.try_pop(item)) {
                    batch.push_back(item);
                } else {
                    break;
                }
            }
            
            if (batch.empty()) {
                // Queue empty - check if we should terminate
                if (pq.empty() && active_threads <= 1) {
                    break;
                }
                this_thread::sleep_for(chrono::microseconds(10));
                continue;
            }
            
            // Process each item in the batch
            for (const auto& item : batch) {
                if (solution_found.load()) break;
            
            int currentIdx = item.stateIdx;
            
            // Get current compact state
            CompactState currentCompact;
            {
                lock_guard<mutex> lock(states_mtx);
                if (currentIdx >= static_cast<int>(states.size())) {
                    continue;
                }
                currentCompact = states[currentIdx];
            }
            
            // Check if solved
            if (isSolvedCompact(currentCompact)) {
                solution_found = true;
                solution_idx = currentIdx;
                break;
            }
            
            // Decompress and compute reachable positions
            State currentState = decompressState(currentCompact);
            ReachableInfo reach = computeReachable(currentState);
            if (reach.startIndex == -1) {
                continue;
            }
            
            // Collect candidates
            vector<CompactState> newStates;
            vector<int> newIndices;
            vector<char> newDirs;
            vector<int> newHeuristics;
            
            for (int y = 0; y < ROWS; ++y) {
                for (int x = 0; x < COLS; ++x) {
                    int idx = y * COLS + x;
                    if (reach.parent[idx] == -1) {
                        continue;
                    }
                    
                    string movePath;
                    if (idx != reach.startIndex) {
                        movePath = reconstructPath(idx, reach);
                    }
                    
                    State afterWalk;
                    if (!movePath.empty()) {
                        if (!applyMoves(currentState, movePath, afterWalk)) {
                            continue;
                        }
                    } else {
                        afterWalk = currentState;
                    }
                    
                    for (int dir = 0; dir < 4; ++dir) {
                        int by = y + dy[dir];
                        int bx = x + dx[dir];
                        int ty = by + dy[dir];
                        int tx = bx + dx[dir];
                        if (!isWithin(by, bx) || !isWithin(ty, tx)) {
                            continue;
                        }
                        char boxTile = afterWalk.board[by][bx];
                        if (boxTile != 'x' && boxTile != 'X') {
                            continue;
                        }
                        char targetTile = afterWalk.board[ty][tx];
                        if (targetTile != ' ' && targetTile != '.') {
                            continue;
                        }
                        
                        State pushed;
                        if (!tryMove(afterWalk, dir, pushed)) {
                            continue;
                        }
                        // Immediate prune (only when deadlock checks are enabled):
                        // if the pushed box lands on a precomputed simple deadlock cell (and not a target), skip
                        if (enableDeadCheck) {
                            if (!targetMap[ty][tx] && deadCellMap[ty][tx]) {
                                continue;
                            }
                        }
                        if (enableDeadCheck && isDeadState(pushed)) {
                            continue;
                        }
                        
                        // Compress to CompactState
                        CompactState pushedCompact = compressState(pushed);
                        
                        // Check visited - TBB map is thread-safe!
                        if (visited.find(pushedCompact) != visited.end()) {
                            continue;
                        }
                        
                        int h = calculateHeuristicCompact(pushedCompact);
                        newStates.push_back(pushedCompact);
                        newIndices.push_back(idx);
                        newDirs.push_back(directions[dir]);
                        newHeuristics.push_back(h);
                    }
                }
            }
            
            // Add new states
            int currentG;
            {
                lock_guard<mutex> lock(states_mtx);
                if (currentIdx >= static_cast<int>(gScore.size())) {
                    continue;
                }
                currentG = gScore[currentIdx];
                
                for (size_t i = 0; i < newStates.size(); ++i) {
                    // Try to insert - TBB map handles concurrency!
                    auto result = visited.insert(make_pair(newStates[i], static_cast<int>(states.size())));
                    if (!result.second) {
                        // Already visited
                        continue;
                    }
                    
                    states.push_back(newStates[i]);
                    parents.push_back(currentIdx);
                    pushDirs.push_back(newDirs[i]);
                    prePushIndices.push_back(newIndices[i]);
                    
                    int newIndex = static_cast<int>(states.size()) - 1;
                    int newG = currentG + 1;
                    gScore.push_back(newG);
                    
                    if (isSolvedCompact(newStates[i])) {
                        solution_found = true;
                        solution_idx = newIndex;
                        break;
                    }
                    
                    int newF = newG + newHeuristics[i];
                    pq.push({newIndex, newF});
                }
            }
            
            if (solution_found) {
                break;
            }
            }  // End of batch processing loop
        }
        
        active_threads--;
    };
    
    // Launch worker threads (6 threads as per assignment requirements)
    const int num_threads = 6;
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    
    // Build solution if found
    if (solution_found && solution_idx >= 0) {
        return buildSolutionPathCompact(solution_idx, parents, pushDirs, prePushIndices, states);
    }
    
    return "";
}

string solveRegular(const State &initialState) {
    // Use TBB concurrent version for better parallelization
    return solveWithConcurrentBFS(initialState, true);
}

string solveSpecialTerrain(const State &initialState) {
    // Use TBB concurrent version for better parallelization
    return solveWithConcurrentBFS(initialState, false);
}

string solve(const string &filename) {
    State initialState = loadState(filename);
    if (isSolved(initialState)) {
        return "";
    }

    if (hasFragileTile) {
        return solveSpecialTerrain(initialState);
    }
    if (isDeadState(initialState)) {
        return "";
    }
    return solveRegular(initialState);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    string result = solve(argv[1]);
    cout << result << endl;
    return 0;
}
