#pragma GCC diagnostic ignored "-Wc++11-extensions"

#include <array>
#include <tuple>
#include <iostream>
#include <vector>
using namespace std;

struct Position { 
    // x e y son entero de 4 bits (0 hasta el num 16)
    unsigned int x : 4;
    unsigned int y : 4;
};

struct Move {
    Position from;
    Position to;
};

vector<Position> pos_range(int x, int y, int lim, int stp, bool isXcoord) {
    vector<Position> output;
    
    if (isXcoord == true) {
        if (x > lim) {
            for (int i = x; i > lim; i += stp) {
                Position to_append = {i, y};
                output.push_back(to_append);
            }
        } else {
            for (int i = x; i < lim; i += stp) {
                Position to_append = {i, y};
                output.push_back(to_append);
            }
        }
    } else {
        if (y > lim) {
            for (int i = y; i > lim; i += stp) {
                 Position to_append = {x, i};
                output.push_back(to_append);
            }
        } else {
            for (int i = y; i < lim; i += stp) {
                Position to_append = {x, i};
                output.push_back(to_append);
            }
        }
    };
    return output;
}

vector<Move> action_space(int SIZE) {
    vector<Move> all_actions;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            vector<Position> x1 = pos_range(i+1, j, SIZE, 1, true);
            vector<Position> x2 = pos_range(i-1, j, -1, -1, true);
            vector<Position> y1 = pos_range(i, j+1, SIZE, 1, false);
            vector<Position> y2 = pos_range(i, j-1, -1, -1, false);
            x1.insert(std::end(x1), std::begin(x2), std::end(x2));
            x1.insert(std::end(x1), std::begin(y1), std::end(y1));
            x1.insert(std::end(x1), std::begin(y2), std::end(y2));
            while(!x1.empty()) {
                Position _to = x1.back();
                x1.pop_back();
                Position _from = {i,j};
                Move action = {_from, _to};
                all_actions.push_back(action);
            }
        }
    }
    return all_actions;
}


class Tafl {
public:
    array<array<char,9>, 9> board;
    bool currentPlayer;
    bool winner;
    bool turn;
    bool done;

    Tafl (){
        array<array<char,9>, 9> board = {{ {'c',' ',' ','m','m','m',' ',' ','c'}, 
                                           {' ',' ',' ',' ','m',' ',' ',' ',' '}, 
                                           {' ',' ',' ',' ','s',' ',' ',' ',' '},
                                           {'m',' ',' ',' ','s',' ',' ',' ','m'}, 
                                           {'m','m','s','s','r','s','s','m','m'},
                                           {'m',' ',' ',' ','s',' ',' ',' ','m'}, 
                                           {' ',' ',' ',' ','s',' ',' ',' ',' '}, 
                                           {' ',' ',' ',' ','m',' ',' ',' ',' '}, 
                                           {'c',' ',' ','m','m','m',' ',' ','c'}}};
        currentPlayer = 0;
        winner = false;
        done = false;
        turn = 0;
    }
    Tafl (array<array<char,9>, 9> _board, bool currentPlayer, bool winner, short turn, bool done) {
        board=_board;
        currentPlayer = currentPlayer;
        winner = winner;
        turn = turn;
        done = done;
    }
    vector<Position> _pieces(bool currentPlayer) {
        vector<Position> output;
        int rowN = 0;
        for (array<char,9> row : this->board) {
            int colN = 0;
            for(char piece : row) {
                if ((piece == 's' || piece == 'r') && currentPlayer) {
                    Position x = {rowN, colN};
                    output.push_back(x);
                }
                else if (piece == 'm' && !currentPlayer) {
                    Position x = {rowN, colN};
                    output.push_back(x);
                }
                colN++;
            }
            rowN++;
        }
        return output;
    }

    vector<Move> _collisions(Position piece, vector<char> pieces) {
        vector<Move> output;
        char currentPiece = this->board[piece.x][piece.y];
        for (auto pos : pos_range(piece.x, piece.y+1, 9, 1, false)) {
            if (std::find(pieces.begin(), pieces.end(), this->board[pos.x][pos.y]) != pieces.end()) { 
                Move validMove = {piece,pos};
                output.push_back(validMove);
            } else break;
                   
        }
        for (auto pos : pos_range(piece.x, piece.y-1, -1, -1, false)) {
            if (std::find(pieces.begin(), pieces.end(), this->board[pos.x][pos.y]) != pieces.end()) { 
                Move validMove = {piece,pos};
                output.push_back(validMove);
            } else break;
                    
        }
        for (auto pos : pos_range(piece.x+1, piece.y, 9, 1, true)) {
            if (std::find(pieces.begin(), pieces.end(), this->board[pos.x][pos.y]) != pieces.end()) { 
                Move validMove = {piece,pos};
                output.push_back(validMove);
            } else break;        
        }
        for (auto pos : pos_range(piece.x-1, piece.y, -1, -1, true)) {
            if (std::find(pieces.begin(), pieces.end(), this->board[pos.x][pos.y]) != pieces.end()) { 
                Move validMove = {piece,pos};
                output.push_back(validMove);
            } else break;        
        }
        return output;
    } 

    vector<Move> _available_moves(Position piece) {
        char pieceAsChar = this->board[piece.x][piece.y];
        vector<Move> output;
        if (pieceAsChar == ' ' || pieceAsChar == 'c' || pieceAsChar == 't') {
            return output;
        }
        if (pieceAsChar == 's' || pieceAsChar == 'm') {
            vector<char> pieces;
            pieces.push_back(' ');
            return _collisions(piece, pieces);
        } else {
            vector<char> pieces;
            pieces.push_back(' ');
            pieces.push_back('c');
            return _collisions(piece, pieces);
        }


    }  


};


    
    


int main() {
    // Tafl b;
    array<array<char,9>, 9> xboard = {{{'c',' ',' ','m','m','m',' ',' ','c'}, {' ',' ',' ',' ','m',' ',' ',' ',' '}, {' ',' ',' ',' ','s',' ',' ',' ',' '},{'m',' ',' ',' ','s',' ',' ',' ','m'}, {'m','m','s','s','r','s','s','m','m'}, {'m',' ',' ',' ','s',' ',' ',' ','m'}, {' ',' ',' ',' ','s',' ',' ',' ',' '}, {' ',' ',' ',' ','m',' ',' ',' ',' '}, {'c',' ',' ','m','m','m',' ',' ','c'}}};
    Tafl b (xboard,0,0,0,0);
    vector<char> x;

  
    for (int i = 0; i < b.board.size(); i++){
        
        for(int j = 0; j < b.board[0].size(); j++) {
            //cout << "A board: " <<  a.board[i][j] << "\n";
            cout << "|" << b.board[i][j];
        
            bool equal = b.board[i][j] == (char) 1;
        };
        cout << "|" << "\n";
    };
}