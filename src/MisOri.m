function com_ang(fname1,fname2,mtex_path)
    % adding path of MTEX
    if nargin < 3
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
    ebsd = EBSD.load(fname1,CS,'interface','ang',...
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
    
    
    %% Import the Data
    
    % create an EBSD variable containing the data
    ebsd2 = EBSD.load(fname2,CS,'interface','ang',...
      'convertEuler2SpatialReferenceFrame','setting 1');
    [grains2,ebsd2.grainId] = calcGrains(ebsd2('indexed'),'angle',15*degree);
    % clean data noise
    grains2 = smooth(grains2,3);
    ebsd2(grains2(grains2.grainSize<4)) = [];
    ebsd2('notIndexed')=[];
    F = halfQuadraticFilter;
    F.alpha = 10;
    ebsd2 = smooth(ebsd2('indexed'),F,'fill',grains2);
    [grains2,ebsd2.grainId] = calcGrains(ebsd2('indexed'),'angle',15*degree);
    grains2 = smooth(grains2,3);
    
    % Extract the orientation data
    ori1 = ebsd.orientations;
    ori2 = ebsd2.orientations;

    % Calculate the misorientation between corresponding orientations
    %misorientations = misorientation(ori1, ori2);

    % Compute misorientation angles in degrees
    misorientationAngles = angle(ori1, ori2) / degree;
    misorientationAngles(misorientationAngles<15)=0.0;
    counts = 100-sum(misorientationAngles>=15)*100/numel(misorientationAngles);

    % Plotting misorientation angle map
    figure;
    plot(ebsd, misorientationAngles, 'micronbar', 'off');
    colormap(turbo);
    caxis([0 90]);  % Set the color axis from 0 to 90 degrees
    mtexColorbar('title', 'Misorientation Angle (°)');
    saveas(gcf,extractBefore(fname1, '.')+"_acc"+num2str(counts)+"_IPFcom.jpg")
    disp('Saving the IPF map...')
    close(gcf)


end
