# TicTacToe Game using the Mini-Max Algorithm
class TicTacToe:
    def __init__(self, player):
        self.player = player
        self.computer = "O" if player == "X" else "X"
        self.board = [[" " for _ in range(3)] for _ in range(3)]

    def printBoard(self):
        print("-------------")
        for row in self.board:
            print("| " + " | ".join(row) + " |")
            print("-------------")
        print()
    
    def checkWinner(self, player:str):
        # Either win horizontally
        for i in range(3):
            if(self.board[i][0] == self.board[i][1] == self.board[i][2] == player):
                return True
        # Or win vertically
        for j in range(3):
            if(self.board[0][j] == self.board[1][j] == self.board[2][j] == player):
                return True
            
        # Or win diagonally
        if(self.board[0][0] == self.board[1][1] == self.board[2][2] == player):
            return True
        elif(self.board[0][2] == self.board[1][1] == self.board[2][0] == player):
            return True
        # If no winning has happened
        return False

    def evaluate(self):
        if self.checkWinner(self.player): 
            return -1
        elif self.checkWinner(self.computer):
            return 1
        return 0
    
    def isFull(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == " ":
                    return False
        return True

    def minimax(self, player_type:str):
        # Check Base Case --> leaf (somebody has won or draw)
        score = self.evaluate()
        if score !=0: # Someone has won
            return score
        if self.isFull(): # All slots filled but no winner (draw)
            return 0
        
        if player_type == "max": # Maximizing player --> Computer
            best_score = -float("inf") # Since maximizing, consider the smallest value
            # All empty slots can be targetted for simulation
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == " ":
                        self.board[i][j] = self.computer
                        value = self.minimax(player_type="min") # Simulate minimizer's move
                        self.board[i][j] = " " # Reset the board after simulation is complete
                        if(value > best_score):
                            best_score = value
            return best_score
        
        else: # Minimizing Player --> Human
            best_score = float("inf") # Since minimizing, consider the largest value
            # All empty slots can be targetted for simulation
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == " ":
                        self.board[i][j] = self.player
                        value = self.minimax(player_type="max") # Simulate maximizer's move
                        self.board[i][j] = " " # Reset the board after simulation is complete
                        if(value < best_score):
                            best_score = value
            return best_score

    def bestMoveForComputer(self):
        # Find the best move for the computer given the current state of board via simulation
        best_score = -float("inf")
        move = (-1,-1)
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == " ":
                    self.board[i][j] = self.computer
                    value = self.minimax(player_type="min") # Recursively simulate next moves
                    self.board[i][j] = " " # Reset the board after simulation is complete
                    if(value > best_score):
                        best_score = value
                        move = (i,j)
        return move
    
    def getUserInput(self):
        while True:
            try:
                row = int(input("Enter row (1-3): ")) - 1
                col = int(input("Enter col (1-3): ")) - 1

                if row not in [0,1,2] or col not in [0,1,2]:
                    print("Please enter numbers between 1 and 3.\n")
                    continue

                if self.board[row][col] != " ":
                    print("That spot is already taken. Try again.\n")
                    continue

                return row, col
            except ValueError:
                print("Invalid input! Please enter numbers only.\n")
        

def playGame():
    player_choice = input("Enter Your Choice among X and O: ")
    computer_choice = "X" if player_choice=="O" else "O"
    print(f"Tic-Tac-Toe (You = {player_choice}, Computer = {computer_choice})")
    game = TicTacToe(player=player_choice)
    game.printBoard()

    while True:
        # Player's Move
        row, col = game.getUserInput()
        game.board[row][col] = game.player

        if game.checkWinner(game.player):
            game.printBoard()
            print("Player wins!")
            break
        if game.isFull():
            game.printBoard()
            print("It's a draw!")
            break

        # Computer's move
        i, j = game.bestMoveForComputer()
        game.board[i][j] = game.computer

        game.printBoard()
        if game.checkWinner(game.computer):
            print("Computer wins!")
            break
        if game.isFull():
            print("It's a draw!")
            break


if __name__ == "__main__":
    playGame()