#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

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

void computeDeadCells() {
    deadCellMap.assign(ROWS, vector<bool>(COLS, false));
    if (ROWS == 0 || COLS == 0) {
        return;
    }

    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (baseBoard[y][x] == '#') {
                continue;
            }
            if (targetMap[y][x]) {
                continue;
            }
            bool up = isWallForBox(y - 1, x);
            bool down = isWallForBox(y + 1, x);
            bool left = isWallForBox(y, x - 1);
            bool right = isWallForBox(y, x + 1);
            if ((up && left) || (up && right) || (down && left) || (down && right)) {
                deadCellMap[y][x] = true;
            }
        }
    }

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
    computeDeadCells();

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

bool isDeadState(const State &state) {
    for (int y = 0; y < static_cast<int>(state.board.size()); ++y) {
        for (int x = 0; x < static_cast<int>(state.board[y].size()); ++x) {
            if (state.board[y][x] == 'x') {
                if (y < static_cast<int>(deadCellMap.size()) && x < static_cast<int>(deadCellMap[y].size()) &&
                    deadCellMap[y][x]) {
                    return true;
                }
                if (baseBoard[y][x] == '@') {
                    return true;
                }
            }
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
        current = std::move(next);
    }
    out = std::move(current);
    return true;
}

string solveWithBFS(const State &initialState, bool enableDeadCheck) {
    if (enableDeadCheck && isDeadState(initialState)) {
        return "";
    }

    queue<State> todo;
    unordered_map<State, string, StateHash> visited;
    visited.reserve(8192);

    todo.push(initialState);
    visited.emplace(initialState, "");

    while (!todo.empty()) {
        State current = std::move(todo.front());
        todo.pop();
        const string &currPath = visited.find(current)->second;

        ReachableInfo reach = computeReachable(current);
        if (reach.startIndex == -1) {
            continue;
        }

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
                    if (!applyMoves(current, movePath, afterWalk)) {
                        continue;
                    }
                } else {
                    afterWalk = current;
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
                    if (enableDeadCheck && isDeadState(pushed)) {
                        continue;
                    }

                    string newPath = currPath;
                    newPath += movePath;
                    newPath.push_back(directions[dir]);

                    if (isSolved(pushed)) {
                        return newPath;
                    }

                    auto [it, inserted] = visited.emplace(pushed, newPath);
                    if (inserted) {
                        todo.push(std::move(pushed));
                    }
                }
            }
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
