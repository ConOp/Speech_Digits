recordPath = fullfile('Digit_Dataset','recordings');
datastore = audioDatastore(recordPath);
datastore.Labels = label_distributer(datastore);

rng default;
datastore = shuffle(datastore);
[dataTrain,dataTest] = splitEachLabel(datastore,0.8);

feature_array = cell(numel(dataTrain.Files),1);
fprintf("Extracting Features!\n");
for i=1:size(dataTrain.Files)
    [audioIn,fs] = audioread(dataTrain.Files{i});
    %These stuff below are the features
    aFE = audioFeatureExtractor(...
    "SampleRate",fs, ...
    "Window",hamming(round(0.2*fs),"periodic"), ...
    "OverlapLength",round(0.02*fs), ...
    "mfcc",true, ...
    "mfccDelta",true, ...%turned off
    "mfccDeltaDelta",true, ...%turned off
    "pitch",true, ...
    "spectralCentroid",true);%turned off
    
    feature_array{i} = extract(aFE,audioIn);
    
    fprintf("Done: "+i+"\n");
end

sum_rows = 0;

for i=1:size(feature_array)
   [rows, col] = size(feature_array{i});
   sum_rows= sum_rows + rows;
end

labels = zeros(sum_rows,1);
cell_array = vertcat(feature_array{:});
iskip=1;

for i=1:size(feature_array)
    [rows, col] = size(feature_array{i});
    for j =1:rows
        labels(iskip) = double(string(dataTrain.Labels(i)));
        iskip=iskip+1;
    end
   
end

fprintf("Training Model\n");
model = fitcecoc(cell_array,labels);
saveLearnerForCoder(model, 'newfinalmodel'); % used to save a trained model

