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

using namespace std;

// Debug flag: set to true to monitor thread workload (disable for production)
#define DEBUG_THREAD_LOAD false

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
        size_t hash = 0;
        for (const auto &row : state.board) {
            for (char c : row) {
                hash = hash * 131 + static_cast<unsigned char>(c);
            }
        }
        hash = hash * 131 + static_cast<size_t>(state.player_y);
        hash = hash * 131 + static_cast<size_t>(state.player_x);
        return hash;
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

// Check if position contains wall or box
inline bool isWallOrBox(const State &state, int y, int x) {
    if (!isWithin(y, x)) return true;
    char c = state.board[y][x];
    return c == '#' || c == 'x' || c == 'X';
}

// Simple 2x2 deadlock: 4 boxes forming a square, not all on targets
bool has2x2Deadlock(const State &state, int y, int x) {
    if (state.board[y][x] != 'x') return false;
    
    // Only check if this box is top-left of a 2x2
    if (!isWithin(y+1, x+1)) return false;
    
    // Check if we have a 2x2 square of boxes
    bool b00 = (state.board[y][x] == 'x' || state.board[y][x] == 'X');
    bool b01 = (state.board[y][x+1] == 'x' || state.board[y][x+1] == 'X');
    bool b10 = (state.board[y+1][x] == 'x' || state.board[y+1][x] == 'X');
    bool b11 = (state.board[y+1][x+1] == 'x' || state.board[y+1][x+1] == 'X');
    
    // If all 4 positions have boxes
    if (b00 && b01 && b10 && b11) {
        // Count how many are on targets
        int onTarget = 0;
        if (state.board[y][x] == 'X') onTarget++;
        if (state.board[y][x+1] == 'X') onTarget++;
        if (state.board[y+1][x] == 'X') onTarget++;
        if (state.board[y+1][x+1] == 'X') onTarget++;
        
        // If not all on target, count total targets in this 2x2
        if (onTarget < 4) {
            int targets = 0;
            if (targetMap[y][x]) targets++;
            if (targetMap[y][x+1]) targets++;
            if (targetMap[y+1][x]) targets++;
            if (targetMap[y+1][x+1]) targets++;
            
            // Deadlock if less than 4 targets (can't all reach targets)
            if (targets < 4) return true;
        }
    }
    
    return false;
}

// Check if position is a wall (NOT box)
inline bool isWall(int y, int x) {
    if (!isWithin(y, x)) return true;
    return baseBoard[y][x] == '#' || baseBoard[y][x] == '@';
}

// Check if box is frozen by walls only (not other boxes)
bool isFrozenBox(const State &state, int y, int x) {
    if (state.board[y][x] != 'x') return false;
    if (targetMap[y][x]) return false; // Already on target, not a problem
    
    // Check if frozen in BOTH directions by WALLS only (truly stuck permanently)
    bool frozenH = (isWall(y, x-1) && isWall(y, x+1));
    bool frozenV = (isWall(y-1, x) && isWall(y+1, x));
    
    // Only deadlock if frozen in BOTH horizontal AND vertical by walls
    return frozenH && frozenV;
}

// Check if box is in a line deadlock along a wall
bool isLineDeadlock(const State &state, int y, int x) {
    if (state.board[y][x] != 'x') return false;
    if (targetMap[y][x]) return false;
    
    // Check horizontal line along wall
    if (isWallForBox(y-1, x) || isWallForBox(y+1, x)) {
        // Count consecutive boxes along wall
        int left = 0, right = 0;
        int targetsLeft = 0, targetsRight = 0;
        
        // Check left
        for (int dx = -1; x + dx >= 0; dx--) {
            char c = state.board[y][x+dx];
            if (c == 'x') {
                left++;
                if (!targetMap[y][x+dx]) continue;
                else { targetsLeft++; break; }
            } else if (c == 'X') {
                left++; targetsLeft++; break;
            } else break;
        }
        
        // Check right
        for (int dx = 1; x + dx < COLS; dx++) {
            char c = state.board[y][x+dx];
            if (c == 'x') {
                right++;
                if (!targetMap[y][x+dx]) continue;
                else { targetsRight++; break; }
            } else if (c == 'X') {
                right++; targetsRight++; break;
            } else break;
        }
        
        // If blocked on both sides without targets, it's deadlock
        if (left + right >= 1 && targetsLeft == 0 && targetsRight == 0) {
            return true;
        }
    }
    
    // Check vertical line along wall
    if (isWallForBox(y, x-1) || isWallForBox(y, x+1)) {
        int up = 0, down = 0;
        int targetsUp = 0, targetsDown = 0;
        
        // Check up
        for (int dy = -1; y + dy >= 0; dy--) {
            char c = state.board[y+dy][x];
            if (c == 'x') {
                up++;
                if (!targetMap[y+dy][x]) continue;
                else { targetsUp++; break; }
            } else if (c == 'X') {
                up++; targetsUp++; break;
            } else break;
        }
        
        // Check down
        for (int dy = 1; y + dy < ROWS; dy++) {
            char c = state.board[y+dy][x];
            if (c == 'x') {
                down++;
                if (!targetMap[y+dy][x]) continue;
                else { targetsDown++; break; }
            } else if (c == 'X') {
                down++; targetsDown++; break;
            } else break;
        }
        
        if (up + down >= 1 && targetsUp == 0 && targetsDown == 0) {
            return true;
        }
    }
    
    return false;
}

// Check if current state is a deadlock using precomputed simple deadlocks
bool isDeadState(const State &state) {
    for (int y = 0; y < static_cast<int>(state.board.size()); ++y) {
        for (int x = 0; x < static_cast<int>(state.board[y].size()); ++x) {
            if (state.board[y][x] == 'x') {
                // Check if box is in a simple deadlock square (precomputed)
                if (y < static_cast<int>(deadCellMap.size()) && x < static_cast<int>(deadCellMap[y].size()) &&
                    deadCellMap[y][x]) {
                    return true;
                }
                // Check if box is on a fragile tile (@)
                if (baseBoard[y][x] == '@') {
                    return true;
                }
            }
            // Check if box on target is on fragile tile
            if (state.board[y][x] == 'X') {
                if (baseBoard[y][x] == '@') {
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

// A* heuristic: sum of Manhattan distances from each box to nearest target
// Level 2 Push Distance Heuristic
// Considers both box-to-goal distance and player-to-push-position cost
int calculateHeuristic(const State &state) {
    int h = 0;
    
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (state.board[y][x] == 'x') {
                int minCost = INT_MAX;
                
                // For each target, calculate push distance
                for (const auto &[ty, tx] : targetPositions) {
                    // Box-to-target Manhattan distance
                    int boxDist = abs(y - ty) + abs(x - tx);
                    
                    // Find the best push direction and estimate player cost
                    int minPlayerCost = INT_MAX;
                    
                    // Try all 4 push directions
                    for (int dir = 0; dir < 4; dir++) {
                        // Position where player needs to be to push the box
                        int push_from_y = y - dy[dir];
                        int push_from_x = x - dx[dir];
                        
                        // Check if push position is valid
                        if (!isWithin(push_from_y, push_from_x)) continue;
                        if (baseBoard[push_from_y][push_from_x] == '#' || 
                            baseBoard[push_from_y][push_from_x] == '@') continue;
                        
                        // Estimate player cost to reach push position
                        int playerCost = abs(state.player_y - push_from_y) + 
                                        abs(state.player_x - push_from_x);
                        
                        // If player is at the box position, add penalty for moving away
                        if (state.player_y == y && state.player_x == x) {
                            playerCost += 1;
                        }
                        
                        minPlayerCost = min(minPlayerCost, playerCost);
                    }
                    
                    // Total push distance = box distance + weighted player cost
                    // Divide player cost by 2 for better balance (still admissible)
                    int totalCost = boxDist;
                    if (minPlayerCost != INT_MAX) {
                        totalCost += minPlayerCost / 2;
                    }
                    
                    minCost = min(minCost, totalCost);
                }
                
                h += minCost;
            }
        }
    }
    
    return h;
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

// Candidate move structure for parallel processing
struct CandidateMove {
    State afterWalk;
    int idx;
    int dir;
    int by, bx, ty, tx;
};

string solveWithBFS(const State &initialState, bool enableDeadCheck) {
    if (enableDeadCheck && isDeadState(initialState)) {
        return "";
    }

    priority_queue<PQItem, vector<PQItem>, greater<PQItem>> pq;
    vector<State> states;
    vector<int> parents;
    vector<char> pushDirs;
    vector<int> prePushIndices;
    vector<int> gScore;
    states.reserve(8192);
    parents.reserve(8192);
    pushDirs.reserve(8192);
    prePushIndices.reserve(8192);
    gScore.reserve(8192);

    unordered_map<State, int, StateHash> visited;
    visited.reserve(8192);

    states.push_back(initialState);
    parents.push_back(-1);
    pushDirs.emplace_back('?');
    prePushIndices.emplace_back(-1);
    gScore.push_back(0);
    
    int h0 = calculateHeuristic(initialState);
    pq.push({0, h0});
    visited.emplace(initialState, 0);

    while (!pq.empty()) {
        PQItem item = pq.top();
        pq.pop();
        int currentIdx = item.stateIdx;
        ReachableInfo reach = computeReachable(states[currentIdx]);
        if (reach.startIndex == -1) {
            continue;
        }

        // Step 1: Collect all candidate moves
        vector<CandidateMove> candidates;
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
                    if (!applyMoves(states[currentIdx], movePath, afterWalk)) {
                        continue;
                    }
                } else {
                    afterWalk = states[currentIdx];
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
                    
                    candidates.push_back({afterWalk, idx, dir, by, bx, ty, tx});
                }
            }
        }

        // Step 2: Process candidates in parallel with optimized load balancing
        vector<State> validStates;
        vector<int> validIndices;
        vector<char> validDirs;
        vector<int> validHeuristics;
        
        // Use parallel processing if we have enough candidates
        if (candidates.size() >= 8) {
            // Pre-allocate thread-local storage to reduce contention
            const int num_threads = omp_get_max_threads();
            vector<vector<State>> threadStates(num_threads);
            vector<vector<int>> threadIndices(num_threads);
            vector<vector<char>> threadDirs(num_threads);
            vector<vector<int>> threadHeuristics(num_threads);
            
            // Pre-reserve space for each thread
            size_t reserveSize = (candidates.size() / num_threads) + 1;
            for (int t = 0; t < num_threads; t++) {
                threadStates[t].reserve(reserveSize);
                threadIndices[t].reserve(reserveSize);
                threadDirs[t].reserve(reserveSize);
                threadHeuristics[t].reserve(reserveSize);
            }
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                
                #if DEBUG_THREAD_LOAD
                int workCount = 0;
                #endif
                
                // Manual work distribution: interleaved to ensure load balance
                // Each thread processes every Nth candidate (stride = nthreads)
                for (size_t i = tid; i < candidates.size(); i += nthreads) {
                    #if DEBUG_THREAD_LOAD
                    workCount++;
                    #endif
                    const auto& cand = candidates[i];
                    
                    State pushed;
                    if (!tryMove(cand.afterWalk, cand.dir, pushed)) {
                        continue;
                    }
                    if (enableDeadCheck && isDeadState(pushed)) {
                        continue;
                    }
                    
                    // No locking here! Will check visited later during merge
                    int h = calculateHeuristic(pushed);
                    
                    // Store in thread-local storage (no locking needed)
                    threadStates[tid].push_back(pushed);
                    threadIndices[tid].push_back(cand.idx);
                    threadDirs[tid].push_back(directions[cand.dir]);
                    threadHeuristics[tid].push_back(h);
                }
                
                #if DEBUG_THREAD_LOAD
                #pragma omp critical(debug_output)
                {
                    cerr << "Thread " << tid << " processed " << workCount 
                         << " candidates, found " << threadStates[tid].size() 
                         << " valid states" << endl;
                }
                #endif
            }
            
            // Merge all thread results sequentially (once per iteration)
            // Filter out visited states during merge (no locking needed)
            for (int t = 0; t < num_threads; t++) {
                for (size_t i = 0; i < threadStates[t].size(); ++i) {
                    // Skip already visited states
                    if (visited.find(threadStates[t][i]) != visited.end()) {
                        continue;
                    }
                    validStates.push_back(threadStates[t][i]);
                    validIndices.push_back(threadIndices[t][i]);
                    validDirs.push_back(threadDirs[t][i]);
                    validHeuristics.push_back(threadHeuristics[t][i]);
                }
            }
        } else {
            // Sequential processing for small candidate sets
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto& cand = candidates[i];
                
                State pushed;
                if (!tryMove(cand.afterWalk, cand.dir, pushed)) {
                    continue;
                }
                if (enableDeadCheck && isDeadState(pushed)) {
                    continue;
                }
                
                if (visited.find(pushed) != visited.end()) {
                    continue;
                }
                
                int h = calculateHeuristic(pushed);
                
                validStates.push_back(pushed);
                validIndices.push_back(cand.idx);
                validDirs.push_back(directions[cand.dir]);
                validHeuristics.push_back(h);
            }
        }

        // Step 3: Add valid states to global structures (sequential)
        for (size_t i = 0; i < validStates.size(); ++i) {
            // Double-check not visited (might have been added by another thread)
            if (visited.find(validStates[i]) != visited.end()) {
                continue;
            }
            
            states.push_back(validStates[i]);
            parents.push_back(currentIdx);
            pushDirs.push_back(validDirs[i]);
            prePushIndices.push_back(validIndices[i]);

            int newIndex = static_cast<int>(states.size()) - 1;
            int newG = gScore[currentIdx] + 1;
            gScore.push_back(newG);
            
            visited.emplace(states.back(), newIndex);

            if (isSolved(states.back())) {
                return buildSolutionPath(newIndex, parents, pushDirs, prePushIndices, states);
            }

            int newF = newG + validHeuristics[i];
            pq.push({newIndex, newF});
        }
    }
    return "";
}

string solveRegular(const State &initialState) {
    return solveWithBFS(initialState, true);
}

string solveSpecialTerrain(const State &initialState) {
    return solveWithBFS(initialState, false);
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
