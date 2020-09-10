%Run WTLARS v1.0.0-beta
%Author : Ishan Wickramsingha
%Date : 2019/10/18
clear;clc;

data = 'flower_weighted'; 
save_data = true;

Tolerence = 0.001;               %0.075 tolerence
X = 0;                          %Previous Solution
L0_Mode = false;                 %True for L0 or false for L1 Minimization
Mask_Type = 'KP';               %'KP': Kronecker Product, 'KR': Khatri-Rao Product 
GPU_Computing = true;           %If True run on GPU if available
Plot = true;                    %Plot norm of the residual at runtime
Debug_Mode = false;             %Save TLARS variable into a .mat file given in path in debug mode 
Path = '.\example\';            %Path to save all variables in debug mode
Active_Columns_Limit = 1000;   %Limit of active columns (Depends on the GPU)
Iterations = 1e6;              %Maximum Number of iteratons to run
Precision_factor = 10;          %eps*20
str = '';

algorithm = 'WTLARS';
%%
LP = 'L1';
if L0_Mode
    LP = 'L0';
end

result_path = strcat(Path,'\results\');
data_path = strcat(Path,'\data\');

load(strcat(data_path,data,'.mat'));

if save_data || Debug_Mode
    result_path = strcat(result_path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_',datestr(now,'yyyymmdd_HHMM'),'\');
    mkdir(result_path);
    diary(strcat(result_path,algorithm,'_',LP,'_',data,'_',num2str(Active_Columns_Limit),'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'.log'));
    diary on
    %profile on
end

if strcmp(Mask_Type,'KR') 
    product = 'Khatri-Rao';
else
    product = 'Kronecker';
end

fprintf('Running %s %s for %s until norm of the residual reach %d%%  \n\n', algorithm, product, data, Tolerence);
fprintf('Dictionary = %s \n\n',str);

[ X, Active_Columns, x, Parameters, Stat, Ax, X_all ] = WTLARS( Y, D_Cell_Array, w, Tolerence, X, L0_Mode, Mask_Type, GPU_Computing, Plot, Debug_Mode, result_path, Active_Columns_Limit, Iterations, Precision_factor );

%% Test

for n=1:length(D_Cell_Array)
    D_n=D_Cell_Array{n};
    D_Cell_Array(n) = {normc(D_n)};
end

if GPU_Computing && gpuDeviceCount == 0
    GPU_Computing = false;
end

y = normc(vec(Y));
r = y - Ax;

fprintf('\nTLARS Completed. \nNorm of the Residual = %g \n', norm(r));

% figure(3);
% plot(abs(X_all(:,1:100)));

%%
if save_data || Debug_Mode
    diary off
    %profile off
    save(strcat(result_path,algorithm,'_',LP,'_',data,'_Results','.mat'),'Ax','Parameters','Stat','Y','D_Cell_Array','Active_Columns', 'x', 'X','-v7.3');
    save(strcat(result_path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'_GPU.mat'),'-v7.3');    
    
    f = gcf;
    savefig(f,strcat(result_path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'.fig'));
    saveas(f,strcat(result_path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'.jpg'));
    %profsave(profile('info'),strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS')));
end



