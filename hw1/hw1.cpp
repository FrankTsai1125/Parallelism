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

// State 是「一個 Sokoban 局面」的資料容器：包含整張棋盤 board 以及玩家座標 player_y/player_x。
// 後面像 tryMove()、computeReachable()、isDeadState() 都需要把「局面」當成一個整體傳來傳去，
// 所以把相關資料包成一個型別最方便。
struct State {
    vector<string> board; //儲存棋盤，是「當前局面」會變動（玩家走、箱子推）。
    int player_y = 0; //儲存玩家位置
    int player_x = 0;

    bool operator==(const State &other) const {
        return player_y == other.player_y && player_x == other.player_x && board == other.board;
    }
    // bool operator==(const State &other) const：定義「兩個 State 如何判斷相等」。
    // const State &other：用 const reference 避免複製、且不允許改動 other
    // 末尾 const：保證這個比較函式不會修改 *this
    // 回傳條件：玩家座標相同且棋盤內容完全相同才算相等
};

// Compact State: Memory-efficient representation
// Stores only box positions and player position as encoded integers
//TODO
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
//限制可見範圍（internal linkage）：這些變數/函式只在這個 .cpp 檔內可用，外部其他檔案無法 extern 引用到。
//避免命名衝突：如果專案有多個 .cpp，每個檔案都可能有 baseBoard 這種名字；匿名 namespace 能避免連結時撞名。
//等價於 static 全域（但更現代/一致）：在 C++ 中匿名 namespace 是「把東西做成檔案私有」的標準寫法

namespace { //定義全域變數
vector<string> baseBoard; //儲存原始棋盤
vector<vector<bool>> targetMap; //儲存目標位置
vector<vector<bool>> deadCellMap; //儲存死角位置
vector<vector<bool>> goalReachable; //儲存目標位置是否可達
vector<pair<int, int>> targetPositions;
int ROWS = 0;
int COLS = 0;
bool hasFragileTile = false;

// Helper functions for position encoding/decoding
inline uint16_t encodePos(int y, int x) {
    return static_cast<uint16_t>(y * COLS + x);
}
//此函數需要用關鍵字 inline 於欲使用之函數名前方，使 compiler 將函數呼叫替換為函數的本體代碼，
// 而非如普通函數要透過函數呼叫機制（如：跳轉到函數地址並回傳）。
// 這種方式可減少函數呼叫的開銷（如：stack 的 push、pop 操作），從而提高程式執行效率。
inline pair<int, int> decodePos(uint16_t pos) {
    return {pos / COLS, pos % COLS};
}
//只做一件事：判斷座標有沒有在地圖範圍內。
//true：在範圍內
//false：出界
inline bool isWithin(int y, int x) {
    return y >= 0 && y < ROWS && x >= 0 && x < COLS;
}
//回答的是「對箱子而言，這格能不能站」
//如果出界：直接回 true（把出界當牆，這樣 corner/corridor 判斷會更簡單）
//如果在界內：看 baseBoard[y][x] 是不是 # 或 @
//#：牆，箱子不能站
//@：fragile tile，規格要求箱子不能站，所以也當牆
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
    deadCellMap.assign(ROWS, vector<bool>(COLS, false)); //替換或重新初始化，一開始先全部都設定為 false
    goalReachable.assign(ROWS, vector<bool>(COLS, false));
    if (ROWS == 0 || COLS == 0) {
        return;
    }

    // Step 1: Mark corner deadlocks (always safe)
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (baseBoard[y][x] == '#' || targetMap[y][x]) {
                continue; //如果是牆 # 或是目標格 targetMap[y][x]==true → 跳過
            } //檢查上下左右是不是「對箱子而言的牆
            bool up = isWallForBox(y - 1, x);
            bool down = isWallForBox(y + 1, x);
            bool left = isWallForBox(y, x - 1);
            bool right = isWallForBox(y, x + 1);
            // Corner deadlocks
            if ((up && left) || (up && right) || (down && left) || (down && right)) {
                deadCellMap[y][x] = true; //如果上下左右都是「對箱子而言的牆」→ 這格是死角
            }
        }

    // Step 4: Reverse box reachability from targets
    queue<pair<int, int>> q;
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (targetMap[y][x]) {
                goalReachable[y][x] = true; //先把所有 target 格丟進 queue，標 goalReachable=true
                q.emplace(y, x); //In-place Construction
                //效能提升：它避免了 push 操作中可能產生的臨時物件和隨後的拷貝或移動，減少了記憶體分配和資料複製的開銷，提高了效能。
                //push(T object)：將一個已經建構好的 T 型物件傳入，然後複製/移動到佇列中。
            }
        }
    }
    //建立 上下左右 的 陣列
    const int revDy[4] = {-1, 0, 1, 0};
    const int revDx[4] = {0, -1, 0, 1};

    //得到一張 goalReachable，代表「箱子若在這格，理論上有機會（在不考慮其他箱子的阻擋下）被推到某個目標」。
    //後面搜尋時會用它做剪枝（例如推箱目的地 ty,tx 若 !goalReachable[ty][tx] 就直接不擴展）。
    while (!q.empty()) {
        auto [y, x] = q.front();
        q.pop();
        //你現在在 reverse BFS 裡，已知「箱子在 (y,x) 是可以到目標的」。那你要找「箱子上一步可能在哪裡」，也就是 prev：
        for (int dir = 0; dir < 4; ++dir) {
            //箱子從 prev 被推到 (y,x)，所以 prev = (y,x) - direction，-1 是往反方向推
            int prevY = y - revDy[dir];
            int prevX = x - revDx[dir];
            //玩家在那一步推箱時要站的位置，-2 是往反方向推兩格
            int playerY = y - 2 * revDy[dir];
            int playerX = x - 2 * revDx[dir];
            //在 reverse BFS 裡判斷「這個 prev 位置是否合理」，合理就把 goalReachable[prev]=true。這樣後面你就能用 goalReachable[ty][tx] 快速剪枝：推到一個永遠回不到目標的格子就不要推。
            if (!isWithin(prevY, prevX) || !isWithin(playerY, playerX)) {
                continue; //prev 與 player 位置都要在地圖內
            }

            // Box cannot stay on wall or fragile tile
            if (baseBoard[prevY][prevX] == '#' || baseBoard[prevY][prevX] == '@') {
                continue; //prev 位置不能是 # 或 @（箱子不能站牆/fragile）
            }

            // Player must have a valid standing position when pushing
            if (baseBoard[playerY][playerX] == '#') {
                continue; //player 站的位置不能是 #（玩家至少要能站在那裡推）
            }
            //goalReachable 是一張布林表：goalReachable[y][x] == true 表示「箱子站在 (y,x) 這格，有機會（理論上）被推到某個目標格」
            //找出遺漏的 goalReachable，把 prev 位置標成 true
            if (!goalReachable[prevY][prevX]) {
                goalReachable[prevY][prevX] = true;
                q.emplace(prevY, prevX);
            }
        }
    }
    }

    // Step 2: Mark corridor deadlocks (horizontal)
    //把箱子推進「沒有目標的死走廊」，就無法再把箱子推到目標（剪枝用）
    for (int y = 0; y < ROWS; ++y) {
        int x = 0;
        //找每一列中，被牆分隔出來的一段連續區間（baseBoard[y][x] != '#' 的連續片段）
        while (x < COLS) {
            if (baseBoard[y][x] == '#') {
                ++x;
                continue;
            }
            vector<int> cells;
            bool corridor = true;
            bool hasTarget = false;
            //若這一段的每個 cell 都滿足 up && down（上下都是箱子視角的牆/fragile/邊界），就視為「水平走廊」
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
            //若該走廊內 沒有任何 target（hasTarget==false）→ 走廊內所有非 target 的格子標成 deadCellMap=true
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
//TODO
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
    ifstream file(filename); //開啟檔案
    State state; //初始化 State 物件
    baseBoard.clear(); //因為腳本的關係，最好先清空棋盤，避免有殘留的資料
    //是「地圖固定結構」：牆 #、fragile @、地板/目標等，是不會因為走動而改變的底圖。
    //這樣後面像 isWall()、deadlock 判定，就不用每次從動態棋盤猜「這格原本是不是牆/fragile」，
    //直接查 baseBoard 更乾淨。
    targetMap.clear(); //清空目標位置
    deadCellMap.clear(); //清空死角位置
    hasFragileTile = false; //預設為沒有脆弱磚塊

    string line; //讀取檔案的每一行
    int y = 0;
    while (getline(file, line)) {
        state.board.push_back(line); //將每一行加入到棋盤中，原始輸入（含玩家/箱子/牆/目標/fragile）
        string staticLine = line;
        vector<bool> targetRow;
        //把動態角色去掉，產生 staticLine（底圖，不包含玩家/箱子/fragile）
        //顯式型別轉換，size() 回傳 size_t（無號整數），你要存進 int ROWS，所以明確轉成 int
        for (int x = 0; x < static_cast<int>(staticLine.size()); ++x) {
            char ch = line[x];
            char baseChar = ch;
            switch (ch) { //動態物件（玩家/箱子）「剝掉」成靜態底圖（地板），使得 staticLine 最後代表「不會變的底圖」（牆、地板、目標、fragile 等）。
            case 'x':  //A box on a regular tile
            case 'o':  //The player stepping on a regular tile
                baseChar = ' '; //（一般地板）變成空白，看起來沒東西
                break;
            case 'X':  //A box on a target tile
            case 'O':  //The player stepping on a target tile
                baseChar = '.'; //（目標地板）變成小數點，看起來像目標
                break;
            case '!':           // fragile 地板
                baseChar = '@';
                hasFragileTile = true;
                break;
            case '@':          // fragile 地板
                hasFragileTile = true;
                break;
            default:
                break;
            }
            staticLine[x] = baseChar; //staticLine 最後代表「不會變的底圖」（牆、地板、目標、fragile 等）。
            targetRow.push_back(ch == '.' || ch == 'O' || ch == 'X');
            //ch == '.' || ch == 'O' || ch == 'X' 這個運算式的結果是 true/false
            //建立一張「哪些格子是目標點（target）」的布林地圖，
            //後面很多地方要快速查 targetMap[y][x]，
            //不想每次都從字元判斷或掃整張圖。
            //所以 targetMap[y][x] == true 代表 (y,x) 這格是目標格。
            unsigned char lowered = static_cast<unsigned char>(ch);
            //先把 ch 轉成 unsigned char（保證是 0~255）
            if (ch == '!') { //依助教給的樣例，玩家位置在 fragile 地板
                state.player_y = y;
                state.player_x = x;
            } else if (std::tolower(lowered) == 'o') { //玩家位置在一般地板
            //而 tolower() 的目的：讓 o / O 都能被當成玩家（因為 tolower('O') == 'o'）    
                state.player_y = y;
                state.player_x = x;
            }
        }
        baseBoard.push_back(staticLine);
        targetMap.push_back(targetRow);
        ++y; //下一行讀取，目的是為了記錄玩家的y座標
    }

    ROWS = static_cast<int>(baseBoard.size());
    COLS = ROWS > 0 ? static_cast<int>(baseBoard[0].size()) : 0;
    //ternary operator:條件 ? 值1 : 值2，如果條件為 true，則返回值1，否則返回值2
    
    // Precompute target positions for heuristic (must be done before deadlock detection)
    targetPositions.clear();
    //這段在把整張 targetMap 轉成「目標座標列表」targetPositions
    for (int y = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            if (targetMap[y][x]) { //將目標 從 bool 轉成 pair<int,int>
                targetPositions.push_back({y, x}); //將目標位置加入到 targetPositions 中
            }
            //後面算 heuristic（例如箱子到目標的 Manhattan distance、Hungarian/greedy matching）時，
            //需要「所有目標的位置」。用 targetPositions 直接遍歷目標點會比每次都掃整張地圖更方便、也更有效率。
        }
    }
    
    // Compute simple deadlocks using reverse BFS from goals
    computeSimpleDeadlocks();

    return state;
}

bool isSolved(const State &state) {
    //只要棋盤上還存在字元 'x'（箱子在非目標地板），就代表沒解完 → 回傳 false。
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
    if (!isWithin(y, x)) {
        return false;
    }
    char c = state.board[y][x];
    return c == 'x' || c == 'X';
}

bool isStaticSquareDeadlock(const State &state, int top, int left) {
    if (top < 0 || left < 0 || top + 1 >= ROWS || left + 1 >= COLS) {
        return false;
    }

    bool hasNonTargetBox = false;
    int solidCount = 0;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int y = top + dy;
            int x = left + dx;

            bool isWallTile = baseBoard[y][x] == '#' || baseBoard[y][x] == '@';
            bool isTargetTile = targetMap[y][x];
            char c = state.board[y][x];
            bool isBoxTile = (c == 'x' || c == 'X');

            if (isBoxTile && !isTargetTile) {
                hasNonTargetBox = true;
                solidCount++;
            } else if (isWallTile) {
                solidCount++;
            } else if (isBoxTile && isTargetTile) {
                // Box already on target contributes to blocking but shouldn't trigger unless all four are solid
                solidCount++;
            }
        }
    }

    if (!hasNonTargetBox) {
        return false;
    }

    return solidCount == 4;
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

                if (c == 'x') {
                    static const int offsets[4][2] = {{0, 0}, {-1, 0}, {0, -1}, {-1, -1}};
                    for (const auto &off : offsets) {
                        int top = y + off[0];
                        int left = x + off[1];
                        if (isStaticSquareDeadlock(state, top, left)) {
                            return true;
                        }
                    }
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
//TODO
int calculateHeuristicCompact(const CompactState &compact) {
    int n = static_cast<int>(compact.boxes.size());
    int m = static_cast<int>(targetPositions.size());
    
    if (n == 0) return 0;
    
    int matchCost = 0;
    
    // Use Hungarian algorithm for complex cases (5-15 boxes)
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
    
    return matchCost;
}

// Priority queue item for A*
//TODO
struct PQItem {
    int stateIdx;
    int priority; // f = g + h
    
    bool operator>(const PQItem &other) const {
        return priority > other.priority;
    }
};

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

//平行化的 A\* 搜尋 Sokoban 解
// TBB Concurrent BFS: Fully parallelized A* with CompactState and thread-safe containers
string solveWithConcurrentBFS(const State &initialState, bool enableDeadCheck) {
    //如果有開啟剪枝（enableDeadCheck==true），先檢查初始狀態是不是死局 isDeadState(initialState)。
    //如果初始狀態是死局，直接回傳空字串（表示沒解）。
    if (enableDeadCheck && isDeadState(initialState)) {
        return "";
    }

    // TBB concurrent containers - thread-safe without explicit locks!
    tbb::concurrent_priority_queue<PQItem, greater<PQItem>> pq;
    vector<CompactState> states;  //states 存所有已建立的狀態（用 CompactState 省記憶體）。用「索引」代表狀態，方便存 parent/路徑資訊與回朔。
    vector<int> parents; //parents[i] 記錄狀態 i 的父狀態索引。最後找到解時可以回朔整條解路徑。
    vector<char> pushDirs; //pushDirs[i] 記錄從父狀態推到狀態 i 的推箱方向（W/A/S/D 其中一個）。回朔時能把「推的那一步」加進輸出字串。
    vector<int> prePushIndices;//prePushIndices[i] 記錄「在父狀態裡，玩家走到哪個格子後才做這次推箱」的格子 index。朔時能重建「推箱前玩家走路的那段 path」。
    vector<int> gScore;//gScore[i] 記錄從起點到狀態 i 的成本g（你這裡每次推箱讓 g+1）。A* 需要f=g+h；也需要知道目前走了幾步推箱。
    
    // Mutex only for vector operations (vectors are not thread-safe)
    mutex states_mtx;
    
    // TBB concurrent map - no lock needed for most operations!
    //visited 是 thread-safe hash map：key 是 CompactState，value 是該狀態在 states 裡的索引。
    //去重，避免同一個狀態被重複擴展；又能在多執行緒下安全使用。
    tbb::concurrent_unordered_map<CompactState, int, CompactStateHash> visited;
    
    // Atomic flags for synchronization
    //避免 data race；一個 thread 找到解就能通知其他 thread 停止。
    atomic<bool> solution_found(false); //是否已找到解
    atomic<int> solution_idx(-1);  //解所在的狀態索引
    atomic<int> active_threads(0); //目前有多少執行緒在跑
    
    // Initialize with CompactState
    CompactState initialCompact = compressState(initialState);
    //把 initialState 壓縮成 CompactState initialCompact，並把它當作索引 0 放進各種陣列。
    states.push_back(initialCompact);
    parents.push_back(-1); //parents[0]=-1 表示起點沒有父節點
    pushDirs.emplace_back('?'); //prePushIndices[0]=-1 代表起點沒有「推箱動作」與「推前站位」
    prePushIndices.emplace_back(-1);
    gScore.push_back(0);//gScore[0]=0 起點成本為 0
    //concurrent_unordered_map 用法類似 一般unordered_map，但它是 thread-safe 的。
    //visited.insert(make_pair(initialCompact, 0))：把 initialCompact 當作 key，0 當作 value，插入 visited。
    //0 是 initialCompact 在 states 裡的索引，表示起點。
    visited.insert(make_pair(initialCompact, 0)); 
    
    //計算起點 heuristic 
    //A* 的f=g+h 需要 h（g=0，所以 f=h0）。
    int h0 = calculateHeuristicCompact(initialCompact);
    pq.push({0, h0});
    //用途：把起點丟進 priority queue：stateIdx=0、priority=h0。
    //目的：讓 worker threads 之後從 pq 取出起點開始擴展搜尋。
    
    // Worker function for each thread with batch processing for better load balancing
    //宣告一個「工作函式」(lambda) 給每個 thread 跑。
    //[&]：捕捉所有區域變數（包括 active_threads、solution_found、solution_idx、pq、states、parents、pushDirs、prePushIndices、gScore、visited）
    auto worker = [&]() {
        //每個 thread 進來都先 +1 active_threads，後面 queue 空掉時，用它判斷是不是該整體停止。
        active_threads++;
        //.load()：atomic 的安全讀取。
        while (!solution_found.load()) {
            // Try to get a batch of items for better parallelization
            //本次迴圈要處理的一小批 A* 節點。
            vector<PQItem> batch; //每個 thread 處理一小批 A* 節點，減少搶 pq 的 contention。
            const int batch_size = 4;  // Process multiple items per iteration
            //一次最多拿 4 個，減少每次只拿 1 個造成的 contention（搶 pq 太頻繁）。
            
            for (int i = 0; i < batch_size && !solution_found.load(); ++i) {
                PQItem item;
                //嘗試從 pq 拿出目前優先度最高的 item。
                if (pq.try_pop(item)) {
                    //拿得到就回 true，拿不到（queue 暫時空）就回 false，不會卡住等待。
                    batch.push_back(item);
                } else {
                    break;
                }
            }
            //這次完全沒拿到 item。
            if (batch.empty()) {
                // Queue empty - check if we should terminate
                //如果 queue 真的是空的，且活躍執行緒只剩最後一個（或快沒了），表示可能全搜完了 → 退出 worker。
                if (pq.empty() && active_threads <= 1) {
                    break;
                }
                //如果 queue 還有東西，但活躍執行緒只剩最後一個，讓 thread 小睡 10 微秒，避免 CPU 空轉太兇。
                this_thread::sleep_for(chrono::microseconds(10));
                //回到 while 開頭再試一次。
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
                        
                        if (enableDeadCheck && !goalReachable[ty][tx]) {
                            continue;
                        }

                        // EARLY prune: check if target position is a corner deadlock
                        // Only prune corner deadlocks (safest strategy)
                        if (enableDeadCheck && !targetMap[ty][tx]) {
                            bool up = isWall(ty - 1, tx);
                            bool down = isWall(ty + 1, tx);
                            bool left = isWall(ty, tx - 1);
                            bool right = isWall(ty, tx + 1);
                            if ((up && left) || (up && right) || (down && left) || (down && right)) {
                                continue;  // Corner deadlock
                            }
                        }
                        
                        State pushed;
                        if (!tryMove(afterWalk, dir, pushed)) {
                            continue;
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
//遇到 fragile tile 的關卡時，仍然用同一套「平行 A* 搜尋」解，但把第二個參數設成 false。
//false 的意思：關掉某些 deadlock/goalReachable 相關剪枝（也就是「不要做 enableDeadCheck 那套剪枝」），避免在特殊地形規則下誤剪掉正確解。
string solveSpecialTerrain(const State &initialState) {
    // Use TBB concurrent version for better parallelization
    return solveWithConcurrentBFS(initialState, false);
}

string solve(const string &filename) {
    State initialState = loadState(filename);
    if (isSolved(initialState)) {
        return "";
    }
    //如果有脆弱磚塊，就呼叫 solveSpecialTerrain 函數
    if (hasFragileTile) {
        return solveSpecialTerrain(initialState);
    }
    if (isDeadState(initialState)) {
        return "";
    }
    return solveRegular(initialState);
}

int main(int argc, char *argv[]) {
    if (argc != 2) { //執行 ./hw1 01.txt 來測試samples testcases，假如沒有帶入 testcase 則會顯示錯誤
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    //argc[0] 為 ./hw1，argc[1] 為 testcase 檔案名稱
    string result = solve(argv[1]); //呼叫 solve 函數來求解 testcase
    cout << result << endl;
    return 0;
}
