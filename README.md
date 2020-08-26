%WTLARS v1.3.0-beta
%Authors : Ishan Wickramasingha
%Date : 2020/08/26

%MATLAB Version : MATLAB R2017b and above

%% Function Call
%[ X ] = WTLARS( Y, D_cell_array, w, Tolerence);
%[ X, Active_Columns ] = WTLARS( Y, D_cell_array, w, Tolerence, ...);
%[ X, Active_Columns, x ] = WTLARS( Y, D_cell_array, w, Tolerence, ...);
%[ X, Active_Columns, x, Parameters, Stat ] = WTLARS( Y, D_Cell_Array, w, Tolerence, X, L0_Mode, Mask_Type, GPU_Computing, Plot, Debug_Mode, Path, Active_Columns_Limit, Iterations, Precision_factor );

%% Inputs 
%Variable           Type       Default   Description
%Y                  (N-D Array)          = Input data tensor
%D_cell_array       (Cell Array)         = Contains the 1 dimensional dictionarary matrices of the separable dictionary D
%w                  (Numeric Vector)     = Weights as a column vector
%Tolerence          (Numeric)            = The target residual error as a tolerence to stop the algorithm
%X                  (N-D Array)          = Previous partially calculated solution. If X = 0 TLARS runs from the begining.
%L0_mode            (Logical)    False   = True/False : True for L0 or false for L1 Minimization
%Mask_Type          (string/char) 'KP'   = 'KP': Kronecker Product, 'KR': Khatri-Rao Product
%GPU_Computing      (Logical)    True    = True/False : If True run on GPU if available
%Plot               (Logical)    False   = Plot norm(r) at runtime
%Debug_Mode         (Logical)    False   = True/False : Save TLARS variable into a .mat file given in path in debug mode
%Path               (string/char) ''     = Path to save all variables in debug mode
%Active_Columns_Limit(Numeric)1e+6  = Limit of active columns (Depends on the GPU)
%Iterations         (Numeric)    numel(Y)= Maximum Number of iteratons to run
%Precision_factor   (Numeric)  5      = Round to 5 times the machine precission 

%% Outputs  
%
%X              (N-D Array)  = Output coefficients in a Tensor format 
%Active_Columns (Numeric Array)  = Active columns of the dictionary
%x              (Numeric Array)= Output coefficients for active columns in a vector format
%
%Parameters = Algorithm Parameters class
%     Parameters.iterations     = t : Total number of iterations 
%     Parameters.residualNorm   = norm(r) : Norm of the Residual at the final solution
%     Parameters.lambda         = Lambda  : Lambda value at the final solution
%     Parameters.activeColumnsCount = Number of Active Columns
%     Parameters.time           = Total Time spent
%
%Stat = WTLARS Statistics Map for each iteration t
%     Stat(t).iteration     = Iteration t
%     Stat(t).residualNorm  = Norm of the residual at iteration t 
%     Stat(t).column        = Changed column at iteration t  
%     Stat(t).columnIndices = Factor indices of the added column
%     Stat(t).addColumn     = Add a column or remove a column at iteration t   
%     Stat(t).activeColumnsCount = length of the active columns at iteration t 
%     Stat(t).delta         = Delta at iteration t
%     Stat(t).lambda        = Lambda at iteration t
%     Stat(t).time          = Total elapsed time at iteration t
