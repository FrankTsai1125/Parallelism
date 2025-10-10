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
#include <condition_variable>
#include <thread>
#include <chrono>
#include "tbb/concurrent_priority_queue.h"
#include "tbb/concurrent_unordered_map.h"
#include <boost/functional/hash.hpp>

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

// Compact State
struct CompactState {
    vector<uint32_t> boxes;  // use 32-bit for safety
    uint32_t player_pos;     
    
    CompactState() : player_pos(0) {}
    
    bool operator==(const CompactState &other) const {
        return player_pos == other.player_pos && boxes == other.boxes;
    }
};

struct CompactStateHash {
    size_t operator()(const CompactState &state) const {
        size_t seed = 0;
        boost::hash_combine(seed, state.player_pos);
        for (uint32_t box : state.boxes) {
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

inline uint32_t encodePos(int y, int x) {
    return static_cast<uint32_t>(y * COLS + x);
}

inline pair<int, int> decodePos(uint32_t pos) {
    return {static_cast<int>(pos / COLS), static_cast<int>(pos % COLS)};
}
} // namespace

CompactState compressState(const State &state) {
    CompactState compact;
    compact.player_pos = encodePos(state.player_y, state.player_x);
    for (int y = 0; y < static_cast<int>(state.board.size()); ++y) {
        for (int x = 0; x < static_cast<int>(state.board[y].size()); ++x) {
            char c = state.board[y][x];
            if (c == 'x' || c == 'X') {
                compact.boxes.push_back(encodePos(y, x));
            }
        }
    }
    sort(compact.boxes.begin(), compact.boxes.end());
    return compact;
}

State decompressState(const CompactState &compact) {
    State state;
    state.board = baseBoard;
    for (uint32_t boxPos : compact.boxes) {
        auto [y, x] = decodePos(boxPos);
        if (targetMap[y][x]) {
            state.board[y][x] = 'X';
        } else {
            state.board[y][x] = 'x';
        }
    }
    auto [py, px] = decodePos(compact.player_pos);
    state.player_y = py;
    state.player_x = px;
    char &cell = state.board[py][px];
    if (cell == '.') cell = 'O';
    else if (cell == 'X') cell = 'O';
    else if (cell == 'x') cell = 'o';
    else cell = 'o';
    return state;
}

bool isSolvedCompact(const CompactState &compact) {
    for (uint32_t boxPos : compact.boxes) {
        auto [y, x] = decodePos(boxPos);
        if (!targetMap[y][x]) {
            return false;
        }
    }
    return true;
}

inline bool isWithin(int y, int x) {
    return y >= 0 && y < ROWS && x >= 0 && x < COLS;
}

inline bool isWallForBox(int y, int x) {
    if (!isWithin(y, x)) return true;
    char tile = baseBoard[y][x];
    return tile == '#' || tile == '@';
}

void computeSimpleDeadlocks() {
    deadCellMap.assign(ROWS, vector<bool>(COLS, false));
    if (ROWS == 0 || COLS == 0) return;
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (baseBoard[y][x] == '#' || targetMap[y][x]) continue;
            bool up = isWallForBox(y - 1, x);
            bool down = isWallForBox(y + 1, x);
            bool left = isWallForBox(y, x - 1);
            bool right = isWallForBox(y, x + 1);
            if ((up && left) || (up && right) || (down && left) || (down && right)) {
                deadCellMap[y][x] = true;
            }
        }
    }
    // corridor detection skipped for brevity (same as before)
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
        for (int x = 0; x < (int)staticLine.size(); ++x) {
            char ch = line[x];
            char baseChar = ch;
            switch (ch) {
            case 'x':
            case 'o': baseChar = ' '; break;
            case 'X':
            case 'O': baseChar = '.'; break;
            case '!': baseChar = '@'; hasFragileTile = true; break;
            case '@': hasFragileTile = true; break;
            }
            staticLine[x] = baseChar;
            targetRow.push_back(ch == '.' || ch == 'O' || ch == 'X');
            if (ch == '!' || tolower((unsigned char)ch) == 'o') {
                state.player_y = y;
                state.player_x = x;
            }
        }
        baseBoard.push_back(staticLine);
        targetMap.push_back(targetRow);
        ++y;
    }
    ROWS = baseBoard.size();
    COLS = ROWS > 0 ? baseBoard[0].size() : 0;
    targetPositions.clear();
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (targetMap[y][x]) targetPositions.push_back({y, x});
        }
    }
    computeSimpleDeadlocks();
    return state;
}

bool isSolved(const State &state) {
    for (auto &row : state.board)
        for (char c : row)
            if (c == 'x') return false;
    return true;
}

// Additional deadlock helpers
inline bool isWall(int y, int x) {
    if (!isWithin(y, x)) return true;
    return baseBoard[y][x] == '#' || baseBoard[y][x] == '@';
}

bool has2x2Deadlock(const State &state, int y, int x) {
    if (!isWithin(y+1, x+1)) return false;
    bool b00 = (state.board[y][x] == 'x' || state.board[y][x] == 'X');
    bool b01 = (state.board[y][x+1] == 'x' || state.board[y][x+1] == 'X');
    bool b10 = (state.board[y+1][x] == 'x' || state.board[y+1][x] == 'X');
    bool b11 = (state.board[y+1][x+1] == 'x' || state.board[y+1][x+1] == 'X');
    if (b00 && b01 && b10 && b11) {
        int targets = 0;
        if (targetMap[y][x]) targets++;
        if (targetMap[y][x+1]) targets++;
        if (targetMap[y+1][x]) targets++;
        if (targetMap[y+1][x+1]) targets++;
        if (targets < 4) return true;
    }
    return false;
}

bool isFrozenBox(const State &state, int y, int x) {
    if (state.board[y][x] != 'x') return false;
    if (targetMap[y][x]) return false;
    bool frozenH = (isWall(y, x-1) && isWall(y, x+1));
    bool frozenV = (isWall(y-1, x) && isWall(y+1, x));
    return frozenH && frozenV;
}

bool isLineDeadlock(const State &state, int y, int x) {
    if (state.board[y][x] != 'x') return false;
    if (targetMap[y][x]) return false;
    if (isWallForBox(y-1, x) || isWallForBox(y+1, x)) return true;
    if (isWallForBox(y, x-1) || isWallForBox(y, x+1)) return true;
    return false;
}

bool isDeadState(const State &state) {
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            char c = state.board[y][x];
            if (c == 'x' || c == 'X') {
                if (y < (int)deadCellMap.size() && x < (int)deadCellMap[y].size() && deadCellMap[y][x])
                    return true;
                if (baseBoard[y][x] == '@') return true;
                if (has2x2Deadlock(state, y, x)) return true;
                if (isFrozenBox(state, y, x)) return true;
                if (isLineDeadlock(state, y, x)) return true;
            }
        }
    }
    return false;
}

// Hungarian algorithm for min-cost matching
int hungarian(const vector<vector<int>> &cost) {
    int n = cost.size();
    int m = cost[0].size();
    int N = max(n, m);
    vector<vector<int>> a(N, vector<int>(N, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            a[i][j] = cost[i][j];
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
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= N; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    return -v[0];
}

int calculateHeuristicCompact(const CompactState &compact) {
    int n = compact.boxes.size();
    int m = targetPositions.size();
    if (n == 0) return 0;
    vector<vector<int>> cost(n, vector<int>(m, 0));
    for (int i = 0; i < n; ++i) {
        auto [by, bx] = decodePos(compact.boxes[i]);
        for (int j = 0; j < m; ++j) {
            auto [ty, tx] = targetPositions[j];
            cost[i][j] = abs(by - ty) + abs(bx - tx);
        }
    }
    int matchCost = hungarian(cost);
    auto [py, px] = decodePos(compact.player_pos);
    int minPlayer = INT_MAX;
    for (auto boxPos : compact.boxes) {
        auto [by, bx] = decodePos(boxPos);
        minPlayer = min(minPlayer, abs(py - by) + abs(px - bx));
    }
    if (minPlayer == INT_MAX) minPlayer = 0;
    return matchCost + (minPlayer / 2);
}

int calculateHeuristic(const State &state) {
    return calculateHeuristicCompact(compressState(state));
}

// (其餘搜尋演算法保持不變，略去，直接複製你原始檔案即可)
// ... 略 ...

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    string result = solve(argv[1]);
    cout << result << endl;
    return 0;
}
