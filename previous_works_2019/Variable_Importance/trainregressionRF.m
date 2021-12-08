function [RFPackage] = trainregressionRF(DataPackage,iverb)
%function [RFPackage] = trainregressionRFv2(DataPackage,iverb)
% Train a random forecst for regression
% 
% The data is exchanged through a data package structure called
% DataPackage.
% 
% Before calling the function trainregressionRF set up the DataPackage,
% For example as follows
%
%     All input data that will be provided to the random forest, with each
%     variable as a separate column of the matrix InAll
%     DataPackage.InAll=InAll;
%
%     The output variable we will be trying to estimate using the random
%     forest should be placed in OutAll
%     DataPackage.OutAll=OutAll;
%
%     Even though InAll and OutAll should be double precision arrays, if
%     some of the inputs in InAll are actually categorical place the index
%     numbers of these columns in icat, so we can tell the Random Forest
%     that these columns are categorical.
%     DataPackage.icat=icat_indices;
%
%     The names of the input variables should be in a cell array called
%     InputNames
%     DataPackage.InputNames=InputNames;
%
%     The name of the output variable should be in a cell array called
%     OutputNames
%     DataPackage.OutputNames=OutputNames;
%
%     A switch to say if we should randomly split up the data to form an
%     independent validation dataset
%     DataPackage.ivalid=ivalid;
%
%     Set the fraction of the entire dataset to be used for validation
%     DataPackage.validation_fraction=validation_fraction;
%     DataPackage.imissing=imissingrows;
%
%     Set the number of most important variables to include when 
%     creating a horizontal bar graph to provide the ranking of the
%     relative importance of the input variables in estimating the output
%     variable. 
%     DataPackage.ntop=length(DataPackage.InputNames);
%
%     A switch to determine if we want to also estimate the relative
%     importance of the variables using the Relief Algorithm
%     DataPackage.irelieff=0;
%
%     If we want to translate the variable names into a more complete
%     description provide two cell arrays as a dictionary of all possible
%     variable names in VariableName and the corresponding more complete 
%     description in Description
%     DataPackage.VariableName=VariableName;
%     DataPackage.Description=Description;
%
%--------------------------------------------------------------------------
% Find machine specs
[ngpus,ncores,ncpus,archstr,maxsize,endian] = findcapabilitiescomputer();

if ngpus>0
    useGPU='yes';
else
    useGPU='no';
end
if ncpus>1
    useParallel='yes';
else
    useParallel='no';
end

%--------------------------------------------------------------------------
% Make sure DataPackage.PlotsDir directory exists
if ~exist(DataPackage.PlotsDir,'dir')
    if isunix
        command=['mkdir -pv ' DataPackage.PlotsDir];
        system(command);
    else
        mkdir(DataPackage.PlotsDir)
    end
end

HistPlotDir=[DataPackage.PlotsDir '2DHistograms/'];
if ~exist(HistPlotDir,'dir')
    if isunix
        command=['mkdir -pv ' HistPlotDir];
        disp(command);
        system(command);
    else
        mkdir(HistPlotDir)
    end
end

ScatterPlotDir=[DataPackage.PlotsDir 'Scatter/'];
if ~exist(ScatterPlotDir,'dir')
    if isunix
        command=['mkdir -pv ' ScatterPlotDir];
        disp(command);
        system(command);
    else
        mkdir(ScatterPlotDir)
    end
end

TextDir=[DataPackage.TextDir 'Scatter/'];
if ~exist(TextDir,'dir')
    if isunix
        command=['mkdir -pv ' TextDir];
        disp(command);
        system(command);
    else
        mkdir(TextDir)
    end
end

%--------------------------------------------------------------------------
% We will be creating some plots, set up the plot size and the screen 
% location for the first plot
width=600;
height=400;
left=1;
bottom=1;

%--------------------------------------------------------------------------
% figure counter
jfigure=0;

ntop_plot=20;

%--------------------------------------------------------------------------
% Partition data into training and validation datasets

% [mm,nn]=size(In)
% maxcol=max(In)
% if DataPackage.normalize
% for ij=1:nn
%    In(:,ij)=In(:,ij)./max(In(:,ij)); 
%    %pause
%    %max(In(:,ij))
% end
% Out=Out./max(Out);
% end

if DataPackage.ivalid
    tic
    disp('Partitioning data into training and validation datasets.')
    [In,Out,InTest,OutTest] = partitiondataforML(DataPackage.InAll,DataPackage.OutAll,DataPackage.validation_fraction);
    whos In Out InTest OutTest
    toc
   

else
    disp('Use all the data for a training dataset.')    
    In=DataPackage.InAll;
    Out=DataPackage.OutAll;
    whos In Out
end


%--------------------------------------------------------------------------
% Set up the random forest parameters
if isfield(DataPackage,'leaf')
    leaf=DataPackage.leaf;
else
    leaf=1;
end

if isfield(DataPackage,'ntrees')
    ntrees=DataPackage.ntrees;
else
    ntrees=200;
end
    
if isfield(DataPackage,'fboot')
    fboot=DataPackage.fboot;
else
    fboot=1;
end

%--------------------------------------------------------------------------
% Set up the size of the parallel pool
npool=ncores;

%--------------------------------------------------------------------------
% Opening parallel pool
if ncpus>1
    tic
    disp('Opening parallel pool')
    
    % first check if there is a current pool
    poolobj=gcp('nocreate');
    
    % If there is no pool create one
    if isempty(poolobj)
        command=['parpool(' num2str(npool) ');'];
        disp(command);
        eval(command);
    else
        poolsize=poolobj.NumWorkers;
        disp(['A pool of ' poolsize ' workers already exists.'])
    end
    
    % Set parallel options
    paroptions = statset('UseParallel',true);
    toc
    
end

%--------------------------------------------------------------------------
% Training Random Forest
tic
disp('Training Random Forest')
b = TreeBagger(...
        ntrees,In,Out,...
        'Method','regression',...
        'oobvarimp','on',...
        'surrogate','on',...
        'minleaf',leaf,...
        'FBoot',fboot,...
        'cat',DataPackage.icat,...
        'Options',paroptions...
   );
RFPackage.b=b;
toc

%--------------------------------------------------------------------------
% Closing parallel pool
% if ncpus>1
%     tic
%     disp('Closing parallel pool')
%     poolobj = gcp('nocreate');
%     delete(poolobj);            
%     delete(gcp);
%     toc
% end

%--------------------------------------------------------------------------
% Use the fit on training data
tic
disp('Using the fit on training data')
x=Out;
[y,y_std]=predict(b,In);
fit_error=x-y;
cct=corrcoef(x,y);
cct=cct(2,1);
RFPackage.cct=cct;
toc




if DataPackage.ivalid
    % Use the fit on independed validation data
    tic
    disp('Using the fit on independed validation data')
    xv=OutTest;
    [yv,yv_std]=predict(b,InTest);
    ccv=corrcoef(xv,yv);
    ccv=ccv(2,1);
    RFPackage.ccv=ccv;
    toc
end





%--------------------------------------------------------------------------
% Calculate the relative importance of the variables
tic
disp('Calculating the relative importance of the input variables')
oobErrorFullX = b.oobError;
toc

tic
disp('Sorting importance into descending order')
err=b.OOBPermutedVarDeltaError;
RFPackage.RelativeImportance=err;
[B,ierr] = sort(err,'descend');
RFPackage.iRelativeImportance=ierr;
toc

%--------------------------------------------------------------------------
% Out of bag error over the number of grown trees
RFPackage.oobError=oobError(b);


%--------------------------------------------------------------------------
% Ploting how weights change with variable rank
disp('Ploting out of bag error over the number of grown trees')

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

%plot(RFPackage.oobError,'LineWidth',2);
plot(RFPackage.oobError,'-k','LineWidth',2);
xlabel('Number of Trees','FontSize',30)
ylabel('Out of Bag Error','FontSize',30)
title('Out of Bag Error','FontSize',30)
set(gca,'FontSize',16)
set(gca,'LineWidth',2);   
grid on
drawnow
fn=[DataPackage.PlotsDir 'error_number_trees_' DataPackage.OutputNames{1}];
[fnpng]=wrplotpng(fn);
RFPackage.fnoobError=fn;

%--------------------------------------------------------------------------
% Ploting how weights change with variable rank
disp('Ploting how weights change with variable rank')

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

% Calculate the weight gradient and minima location so we know how many
% variables to include
w=RFPackage.RelativeImportance(RFPackage.iRelativeImportance);
fn=[DataPackage.PlotsDir 'variable_weights_' DataPackage.OutputNames{1}];
[RFPackage.iminLocation] = plotWeightFunctionWithGradient(w,'Random Forest',ntop_plot,fn);
RFPackage.fnPlotRelativeImportance=fn;

%--------------------------------------------------------------------------
% Ploting the scatter diagram
disp('Ploting the scatter diagram')

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

plot(x,x,'-b')
hold on
plot(x,y,'ko')
%plot(x,y,'bo')
%errorbar(x,y,y_std,'bo')
if DataPackage.ivalid
    plot(xv,yv,'rs')
    %errorbar(xv,yv,yv_std,'rs')
end
hold off
grid on

if DataPackage.ivalid
    %legend(['1:1'],['Training (#' num2str(length(x)) ')'],['Validation (#' num2str(length(xv)) ')'],'Location','NorthWest')
    legend(['1:1'],['Training'],['Validation'],'Location','NorthWest')
else
    %legend(['1:1'],['Training (#' num2str(length(x)) ')'],'Location','NorthWest')
    legend(['1:1'],['Training '],'Location','NorthWest')
end

px=prctile(x,1:99);
xlim([min(x) px(99)])
ylim([min(x) px(99)])




clear idx description;
idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.OutputNames{1},'_ordinal','')));
description=strtrim(DataPackage.Description{idx});

xlabel(['Actual ' description],'FontSize',30,'Interpreter','latex')
ylabel(['Estimated ' description],'FontSize',30,'Interpreter','latex')
if DataPackage.ivalid
    title(['Estimated ' description ', R$_T$=' num2str(cct,2) ', R$_V$=' num2str(ccv,2)],'FontSize',20,'Interpreter','latex')
else
    title(['Estimated ' description ', R=' num2str(cct,2)],'FontSize',20,'Interpreter','latex')
end
set(gca,'FontSize',16)
set(gca,'LineWidth',2);                
drawnow

fn=[DataPackage.PlotsDir DataPackage.OutputNames{1}];
disp(fn)
[fnpng]=wrplotpng(fn);
RFPackage.fnScatterPlot=fn;

%stop

%--------------------------------------------------------------------------
% Plot 2D histogram training scatter diagram
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>3*width
    left=1;
    bottom=bottom+height;
end

%--------------------------------------------------------------------------
nbins=100;
iwant=find(x>=min(x) & x<=px(95) & y>=min(x) & y<=px(95));
[N,Xedges,Yedges,binX,binY]=histcounts2(x(iwant),y(iwant),nbins);
clear X Y
for j=1:length(Xedges)-1
    X(j)=0.5*(Xedges(j)+Xedges(j+1));
end
for j=1:length(Yedges)-1
    Y(j)=0.5*(Yedges(j)+Yedges(j+1));
end
[XX,YY]=ndgrid(X,Y);

%imagesc([min(X) max(X)],[min(Y) max(Y)],rot90(N));
pcolor(XX,YY,100*N/(sum(sum(N))))
shading interp
colormap jet                
c=caxis;
caxis([c(1) 0.25*c(2)]);
map2 = colormap; map2( 1, : ) = 1; colormap(map2);
colorbar
grid on
xlabel(['Actual ' description],'FontSize',30,'Interpreter','latex')
ylabel(['Estimated ' description],'FontSize',30,'Interpreter','latex')
title('2D Training Scatter Histogram','FontSize',30)
set(gca,'FontSize',16)
set(gca,'LineWidth',2);                
drawnow    

fn=[DataPackage.PlotsDir DataPackage.OutputNames{1} '_scatter_train_hist'];
[fnpng]=wrplotpng(fn);    

%stop
%--------------------------------------------------------------------------
% Plot 2D histogram validation scatter diagram
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>3*width
    left=1;
    bottom=bottom+height;
end

%--------------------------------------------------------------------------
iwant=find(xv>=min(x) & xv<=px(95) & yv>=min(x) & yv<=px(95));
[N,Xedges,Yedges,binX,binY]=histcounts2(xv(iwant),yv(iwant),nbins);
clear X Y
for j=1:length(Xedges)-1
    X(j)=0.5*(Xedges(j)+Xedges(j+1));
end
for j=1:length(Yedges)-1
    Y(j)=0.5*(Yedges(j)+Yedges(j+1));
end
[XX,YY]=ndgrid(X,Y);

%imagesc([min(X) max(X)],[min(Y) max(Y)],rot90(N));
pcolor(XX,YY,100*N/(sum(sum(N))))
shading flat
colormap jet                
c=caxis;
caxis([c(1) 0.25*c(2)]);
map2 = colormap; map2( 1, : ) = 1; colormap(map2);
colorbar
grid on
xlabel(['Actual ' description],'FontSize',30,'Interpreter','latex')
ylabel(['Estimated ' description],'FontSize',30,'Interpreter','latex')
title('2D Validation Scatter Histogram','FontSize',30)
set(gca,'FontSize',16)
set(gca,'LineWidth',2);                
drawnow    

fn=[DataPackage.PlotsDir DataPackage.OutputNames{1} '_scatter_valid_hist'];
[fnpng]=wrplotpng(fn);    

%stop

%--------------------------------------------------------------------------
% Plot the realative importance of the variables
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>3*width
    left=1;
    bottom=bottom+height;
end

%--------------------------------------------------------------------------
% Number of most important variables to plot
ntop=min([ntop_plot DataPackage.ntop]);
weights=err;
iranked=ierr;

%--------------------------------------------------------------------------
% Method Name
MethodName='RandomForest';

% Output File Name
fn=[DataPackage.PlotsDir 'variable_importance_sorted_RandomForest_' DataPackage.OutputNames{1}];

% Output Description
clear idx description;
idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.OutputNames{1},'_ordinal','')));
OutputDescription=strtrim(DataPackage.Description{idx});

fn_txt=[TextDir '/SortedByRF.txt'];
disp(fn_txt)
fid = fopen(fn_txt,'w');
string='Variables sorted by Random Forest Importance';
fprintf(fid,'%s \n',string);
disp(string)

% Set up the descriptions for each bar
for ii=1:ntop
    if (length(DataPackage.Description)>0 & length(DataPackage.VariableName)>0)    
        clear idx description
        disp(DataPackage.InputNames{iranked(ii)})
        idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.InputNames{iranked(ii)},'_ordinal','')))
        description=strtrim(DataPackage.Description{idx(1)});
    else
        description=strrep(DataPackage.InputNames{iranked(ii)},'_','');        
    end
    if ismember(iranked(ii),DataPackage.icat)
        description=[description '$^\dagger$'];
    end
    InputDescriptions{ii}=description;
    
    string=[...
        'Rank ' num2str(ii,'%4u') ', ' ...
        num2str(weights(iranked(ii))) ', ' ...
        description ...
        ];        
    fprintf(fid,'%s \n',string);
    disp(string)            
    
end

fclose(fid)


% Create a ranked and labeled horizontal bar graph of the weights and save to a file
plotRankedList(weights,iranked,ntop,DataPackage.icat,InputDescriptions,OutputDescription,MethodName,fn);
RFPackage.fnBarRelativeImportance=fn;

%stop After Random Forest Rank
%--------------------------------------------------------------------------
if DataPackage.irelieff
    
    % Importance of attributes (predictors) using ReliefF algorithm
    disp('Calculating importance of predictors using Relieff algorithm.')
    tic
    K=10;
    clear weights iranked
    [iranked,weights] = relieff(DataPackage.InAll,DataPackage.OutAll,K,'method','regression');
    toc

    RFPackage.RelativeImportanceRelieff=weights;
    RFPackage.iRelativeImportanceRelieff=iranked;    
    
    %--------------------------------------------------------------------------
    % Ploting how weights change with variable rank
    disp('Ploting how weights change with variable rank')

    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer','painters')
    jfigure=jfigure+1;
    %--------------------------------------------------------------------------
    left=left+width;
    if left>4*width
        left=1;
        bottom=bottom+height;
        if bottom>4*height
            bottom=1;
        end
    end

    % Calculate the weight gradient and minima location so we know how many
    % variables to include
    w=RFPackage.RelativeImportanceRelieff(RFPackage.iRelativeImportanceRelieff);
    fn=[DataPackage.PlotsDir 'variable_weights_relieff_' DataPackage.OutputNames{1}];
    [RFPackage.iminLocationRelieff] = plotWeightFunctionWithGradient(w,'Relieff algorithm',ntop_plot,fn);
    RFPackage.fnPlotRelativeImportanceRelieff=fn;
    
    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer','painters')
    jfigure=jfigure+1;
    
    %--------------------------------------------------------------------------
    left=left+width;
    if left>3*width
        left=1;
        bottom=bottom+height;
    end
    
    %--------------------------------------------------------------------------
    % Method Name
    MethodName='Relieff Algorithm';
    
    % Output File Name
    fn=[DataPackage.PlotsDir 'variable_importance_sorted_Relieff_' DataPackage.OutputNames{1}];
    
    % Output Description
    clear idx description;
    idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.OutputNames{1},'_ordinal','')));
    OutputDescription=strtrim(DataPackage.Description{idx});
    
    fn_txt=[TextDir '/SortedByRelieff.txt'];
    disp(fn_txt)
    fid = fopen(fn_txt,'w');
    string='Variables sorted by Relieff Algorithm';
    fprintf(fid,'%s \n',string);
    disp(string)
    
    % Set up the descriptions for each bar
    for ii=1:ntop
        if (length(DataPackage.Description)>0 & length(DataPackage.VariableName)>0)    
            clear idx description
            idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.InputNames{iranked(ii)},'_ordinal','')))
            description=strtrim(DataPackage.Description{idx(1)});
        else
            description=strrep(DataPackage.InputNames{iranked(ii)},'_','');        
        end
        if ismember(iranked(ii),DataPackage.icat)
            description=[description '$^\dagger$'];
        end
        InputDescriptions{ii}=description;

        string=[...
            'Rank ' num2str(ii,'%4u') ', ' ...
            num2str(weights(iranked(ii))) ', ' ...
            description ...
            ];        
        fprintf(fid,'%s \n',string);
        disp(string)            

    end

    fclose(fid)        
    
    % Create a ranked and labeled horizontal bar graph of the weights and save to a file
    plotRankedList(weights,iranked,ntop,DataPackage.icat,InputDescriptions,OutputDescription,MethodName,fn);
    RFPackage.fnBarRelativeImportanceRelieff=fn;

    %--------------------------------------------------------------------------

end

%--------------------------------------------------------------------------
if ~DataPackage.iInteractionInformation
    
    % Importance of attributes (predictors) using Interaction Information
    disp('Calculating importance of predictors using Interaction Information.')
    tic
    
    %--------------------------------------------------------------------------
    clear features target data_quant
    
    Data=[DataPackage.InAll DataPackage.OutAll];
    
    % For calculating mutual information, continuous variables have to be discretized.
    %data_quant=quantize(Data,'levels',20);

    % Run the default feature selection algorithm, called 'first order utility'. 
    % It is based on approximating the mutual information between the set of selected 
    % variables and the target by expanding interaction information terms of up to degree 2. 
    % In each step, the variable with the highest estimated incremental gain is selected greedily. 
    % The output distinguishes between relevance, i.e., mutual information between a feature and the target; 
    % redundancy, i.e., mutual information between different variables; and conditional redundancy, 
    % which measures the increase of mutual information between the previously selected variables 
    % and the target, conditional on a selected variable.
    ns=size(data_quant);
    ntop_local=ns(2);
    idegree=1;
    features=data_quant(:,1:end-1);
    target=data_quant(:,end);
    [steps,sel_flag,rel,red,cond_red,ilist] = select_features_david(DataPackage.InputNames,features,target,ntop_local,'degree',idegree, 'init', 1:ns(2)-1, 'direction', 'b','prior',1);
    
    % Calculate variable score using relevance and conditional redundancy,
    score=sum([rel; - red; cond_red]);
    RFPackage.RelativeImportanceInteractionInformation=score;
    [B,iscore] = sort(score,'descend');
    RFPackage.iRelativeImportanceInteractionInformation=iscore;    
    
    clear iranked weights
    iranked=iscore;
    weights=score;
    
    fn_txt=[TextDir '/SortedByInteractionInformation.txt'];
    disp(fn_txt)
    fid = fopen(fn_txt,'w');
    string='Variables sorted by InteractionInformation';
    fprintf(fid,'%s \n',string);
    disp(string)
    
    % Relative importance of variables
    for j=1:length(RFPackage.iRelativeImportanceInteractionInformation)
        clear idx description
        %disp(DataPackage.InputNames{RFPackage.iRelativeImportanceInteractionInformation(j)})
        idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.InputNames{RFPackage.iRelativeImportanceInteractionInformation(j)},'_ordinal','')));
        description=char(strtrim(DataPackage.Description{idx(1)}));
        string=['Rank ' num2str(j,'%4u') ', score ' num2str(RFPackage.RelativeImportanceInteractionInformation(RFPackage.iRelativeImportanceInteractionInformation(j)),'%7.3g') ', ' description ];
        fprintf(fid,'%s \n',string);
        disp(string)            

    end
    fclose(fid)       
    
    %--------------------------------------------------------------------------
    % Ploting how weights change with variable rank
    disp('Ploting how weights change with variable rank')

    %--------------------------------------------------------------------------
    figure('Position',[left, bottom, width, height],'Renderer','painters')
    jfigure=jfigure+1;
    %--------------------------------------------------------------------------
    left=left+width;
    if left>4*width
        left=1;
        bottom=bottom+height;
        if bottom>4*height
            bottom=1;
        end
    end

    % Calculate the weight gradient and minima location so we know how many
    % variables to include
    w=RFPackage.RelativeImportanceInteractionInformation(RFPackage.iRelativeImportanceInteractionInformation);
    fn=[DataPackage.PlotsDir 'variable_weights_InteractionInformation_' DataPackage.OutputNames{1}];
    [RFPackage.iminLocationImportanceInteractionInformation] = plotWeightFunctionWithGradient(w,'Interaction Information',ntop_plot,fn);
    RFPackage.fnPlotRelativeImportanceInteractionInformation=fn;
    
    %--------------------------------------------------------------------------
    % Plot bar graph of sorted weights
    figure('Position',[left, bottom, width, height],'Renderer','painters')
    jfigure=jfigure+1;
    
    %--------------------------------------------------------------------------
    left=left+width;
    if left>3*width
        left=1;
        bottom=bottom+height;
    end
    
    %--------------------------------------------------------------------------
    % Method Name
    MethodName='Interaction Information';
    
    % Output File Name
    fn=[DataPackage.PlotsDir 'variable_importance_sorted_InteractionInformation_' DataPackage.OutputNames{1}];
    
    % Output Description
    clear idx description;
    idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.OutputNames{1},'_ordinal','')));
    OutputDescription=strtrim(DataPackage.Description{idx});
    
    % Set up the descriptions for each bar
    for ii=1:ntop
        if (length(DataPackage.Description)>0 & length(DataPackage.VariableName)>0)    
            clear idx description
            idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.InputNames{iranked(ii)},'_ordinal','')));
            description=strtrim(DataPackage.Description{idx});
        else
            description=strrep(DataPackage.InputNames{iranked(ii)},'_','');        
        end
        if ismember(iranked(ii),DataPackage.icat)
            description=[description '$^\dagger$'];
        end
        InputDescriptions{ii}=description;
    end
    
    % Create a ranked and labeled horizontal bar graph of the weights and save to a file
    plotRankedList(weights,iranked,ntop,DataPackage.icat,InputDescriptions,OutputDescription,MethodName,fn);
    RFPackage.fnBarRelativeImportanceInteractionInformation=fn;
    
    %--------------------------------------------------------------------------
    
end

%--------------------------------------------------------------------------
% Rank variables in order of correlation coeffecient and mutual
% information
[r,sr,isort_r,mi,smi,isort_mi]=rankvariables_r_mi(In,Out);

%--------------------------------------------------------------------------
% Ploting how correlation coeffecient change with variable rank
disp('Ploting how mutual information change with variable rank')

%--------------------------------------------------------------------------
% Method Name
MethodName='MutualInformation';

% Output File Name
fn=[DataPackage.PlotsDir 'variable_importance_sorted_MI_' DataPackage.OutputNames{1}];

% Output Description
clear idx description;
idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.OutputNames{1},'_ordinal','')));
OutputDescription=strtrim(DataPackage.Description{idx});

%--------------------------------------------------------------------------
fn_txt=[TextDir '/SortedByMI.txt'];
disp(fn_txt)
fid = fopen(fn_txt,'w');
string='Variables sorted by mutual information';
fprintf(fid,'%s \n',string);
disp(string)
icount=0;
for j=1:length(isort_mi)
    if ~isnan(mi(isort_mi(j)))
        icount=icount+1;
        RFPackage.RelativeImportanceMI(icount)=mi(isort_mi(j));
        RFPackage.iRelativeImportanceMI(icount)=isort_mi(j);        
        clear idx description
        this_name=strrep(DataPackage.InputNames{isort_mi(j)},'_ordinal','');
        idx = find(strcmp(DataPackage.VariableName,this_name));
        description=char(strtrim(DataPackage.Description{idx(1)}));
        string=[...
            'Rank ' num2str(j,'%4u') ' with MI ' num2str(r(isort_mi(j)),'%7.3g') ', ' ...
            this_name ', ' description ...
            ];        
        fprintf(fid,'%s \n',string);
        disp(string)        
        if ismember(isort_mi(j),DataPackage.icat)
            description=[description '$^\dagger$'];
        end
        InputDescriptions{icount}=description;
        
    end
end            
fclose(fid)

%--------------------------------------------------------------------------
% Ploting how correlation coeffecient change with variable rank
disp('Ploting how mutual information change with variable rank')

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

% Calculate the weight gradient and minima location so we know how many
% variables to include
w=RFPackage.RelativeImportanceMI;
fn=[DataPackage.PlotsDir 'variable_weights_MI_' DataPackage.OutputNames{1}];
[RFPackage.iminLocationMI] = plotWeightFunctionWithGradient(w,'Mutual Information',ntop_plot,fn);
RFPackage.fnPlotRelativeImportanceMI=fn;

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

% Create a ranked and labeled horizontal bar graph of the weights and save to a file
fn=[DataPackage.PlotsDir 'variable_importance_sorted_MI_' DataPackage.OutputNames{1}];
plotRankedList(weights,iranked,ntop,DataPackage.icat,InputDescriptions,OutputDescription,MethodName,fn);
RFPackage.fnBarRelativeImportanceMI=fn;

%--------------------------------------------------------------------------
% Method Name
MethodName='CorrelationCoeffecient';

% Output File Name
fn=[DataPackage.PlotsDir 'variable_importance_sorted_R_' DataPackage.OutputNames{1}];

% Output Description
clear idx description;
idx = find(strcmp(DataPackage.VariableName,strrep(DataPackage.OutputNames{1},'_ordinal','')));
OutputDescription=strtrim(DataPackage.Description{idx});

%--------------------------------------------------------------------------
fn_txt=[TextDir '/SortedByR.txt'];
disp(fn_txt)
fid = fopen(fn_txt,'w');
string='Variables sorted by absolute value of correlation coeffecient';
fprintf(fid,'%s \n',string);
disp(string)
icount=0;
for j=1:length(isort_r)
    if ~isnan(r(isort_r(j)))
        icount=icount+1;
        RFPackage.RelativeImportanceR(icount)=r(isort_r(j));
        RFPackage.iRelativeImportanceR(icount)=isort_r(j);        
        clear idx description
        this_name=strrep(DataPackage.InputNames{isort_r(j)},'_ordinal','');
        idx = find(strcmp(DataPackage.VariableName,this_name));
        description=char(strtrim(DataPackage.Description{idx(1)}));
        string=[...
            'Rank ' num2str(j,'%4u') ' with R ' num2str(r(isort_r(j)),'%7.3g') ', ' ...
            this_name ', ' description ...
            ];        
        fprintf(fid,'%s \n',string);
        disp(string)
        if ismember(isort_r(j),DataPackage.icat)
            description=[description '$^\dagger$'];
        end
        InputDescriptions{icount}=description;        
        
    end
end     
fclose(fid)

%--------------------------------------------------------------------------
% Ploting how correlation coeffecient change with variable rank
disp('Ploting how correlation coeffecient change with variable rank')

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

% Calculate the weight gradient and minima location so we know how many
% variables to include
w=RFPackage.RelativeImportanceR;
fn=[DataPackage.PlotsDir 'variable_weights_R_' DataPackage.OutputNames{1}];
[RFPackage.iminLocationR] = plotWeightFunctionWithGradient(w,'Correlation Coeffecient',ntop_plot,fn);
RFPackage.fnPlotRelativeImportanceR=fn;

%--------------------------------------------------------------------------
figure('Position',[left, bottom, width, height],'Renderer','painters')
jfigure=jfigure+1;
%--------------------------------------------------------------------------
left=left+width;
if left>4*width
    left=1;
    bottom=bottom+height;
    if bottom>4*height
        bottom=1;
    end
end

% Create a ranked and labeled horizontal bar graph of the weights and save to a file
fn=[DataPackage.PlotsDir 'variable_importance_sorted_R_' DataPackage.OutputNames{1}];
plotRankedList(weights,iranked,ntop,DataPackage.icat,InputDescriptions,OutputDescription,MethodName,fn);
RFPackage.fnPlotRelativeImportanceR=fn;

%--------------------------------------------------------------------------
if DataPackage.iplotScatter
    plotScatter(DataPackage,RFPackage,ScatterPlotDir);
end

if DataPackage.iplot2DHistograms
    plot2DHistograms(DataPackage,RFPackage,HistPlotDir);
end

if DataPackage.ivalid
    disp(['Random Forest: Rt=' num2str(cct,2) ', Rv=' num2str(ccv,2)])
    RFPackage.Rt=cct;
    RFPackage.Rv=ccv;    
else
    disp(['Random Forest: Rt=' num2str(cct,2)])
    RFPackage.Rt=cct;
end


