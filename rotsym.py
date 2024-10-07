from sympy import symbols, Matrix, BlockMatrix, MatrixSymbol, ZeroMatrix, simplify

# Define symbols
alpha = symbols('alpha', real=True)
R_w2c = MatrixSymbol('R_w2c', 3, 3)
R_c = MatrixSymbol('R_c', 3, 3)
t_w2c = MatrixSymbol('t_w2c', 3, 1)
t_c = MatrixSymbol('t_c', 3, 1)

# Define matrices using Matrix class
T_w2c = BlockMatrix([[R_w2c, t_w2c], [ZeroMatrix(1, 3), Matrix([[1]])]])
T_c = BlockMatrix([[R_c, alpha * t_c], [ZeroMatrix(1, 3), Matrix([[1]])]])

# Alpha-scaled translation component of T_w2c
T_c2w = BlockMatrix([[R_w2c.T, -R_w2c.T * t_w2c], [ZeroMatrix(1, 3), Matrix([[1]])]])

# Construct the expression
T_o = T_c2w * T_c * T_w2c * (T_c * T_w2c)

# Simplify the expression
T_o = simplify(T_o)

# print(T_o)
print(T_o)
