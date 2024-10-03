function MTEX_ana(fname,mtex_path)
    % adding path of MTEX
    if nargin < 2
	    mtex_path="/local/home/jiatang/Documents/mtex-5.9.0";
    end
    addpath(mtex_path)
    startup_mtex
    
    %% Specify Crystal and Specimen Symmetries
    % crystal symmetry
    CS = {... 
      'notIndexed',...
      crystalSymmetry('m-3m', [4 4 4], 'mineral', 'Nickel', 'color', [0.53 0.81 0.98])};
    
    % plotting convention
    setMTEXpref('xAxisDirection','east');
    setMTEXpref('zAxisDirection','outOfPlane');
    
    %% Import the Data
    
    % create an EBSD variable containing the data
    ebsd = EBSD.load(fname,CS,'interface','ang',...
      'convertEuler2SpatialReferenceFrame','setting 1');
    [grains,ebsd.grainId] = calcGrains(ebsd('indexed'),'angle',15*degree);
    % clean data noise
    grains = smooth(grains,3);
    ebsd(grains(grains.grainSize<4)) = [];
    ebsd('notIndexed')=[];
    F = halfQuadraticFilter;
    F.alpha = 10;
    ebsd = smooth(ebsd('indexed'),F,'fill',grains);
    [grains,ebsd.grainId] = calcGrains(ebsd('indexed'),'angle',15*degree);
    grains = smooth(grains,3);
    
    %% plot EBSD IPF map
    figure
    ipfKey = ipfColorKey(ebsd('indexed'));
    ipfKey.inversePoleFigureDirection = vector3d.Z;   % colors follow orientations in y direction
    colors = ipfKey.orientation2color(ebsd('indexed').orientations);
    plot(ebsd('indexed'),colors)
    hold on
    plot(grains.boundary,'linewidth',1.5)
    hold off
    saveas(gcf,[fname(1:end-4) '_IPF.jpg'])
    disp('Saving the IPF map...')
    close(gcf)
    
    %% grain analysis and save grain information into csv file
    % fit the ellipse
    [omega,a,~] = fitEllipse(grains);
    grain_diameter = 2*grains('indexed').equivalentRadius;
    Aspect_ratio = grains('indexed').aspectRatio;
    Ellipse_LongAxis = a;
    Ellipse_Angle = omega;
    Matr=[grain_diameter Ellipse_LongAxis Aspect_ratio Ellipse_Angle*180./pi];
    Matr=["grain_diameter" "Ellipse_LongAxis" "Aspect_ratio" "Ellipse_Angle";Matr];
    writematrix(Matr,[fname(1:end-4) '_grain.csv']);
    
    psi = calcKernel(grains.meanOrientation);
    odf = calcDensity(ebsd.orientations,'kernel',psi);
    h=[Miller(0,0,1,ebsd.CS),Miller(1,1,1,ebsd.CS),Miller(1,1,0,ebsd.CS)];
    figure
    plotPDF(odf,h,'upper','projection','eangle')
    CLim(gcm, [0, 2.2]);  % Set the same color limits for all plots
    colorbar;
    set(gcf, 'Position', [100, 100, 900, 300]);
    saveas(gcf,[fname(1:end-4) '_PF.jpg'])
    disp('Saving the IPF map...')
    close(gcf)
    

end
