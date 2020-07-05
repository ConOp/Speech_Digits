recordPath = fullfile('Digit_Dataset','recordings');
datastore = audioDatastore(recordPath);
datastore.Labels = label_distributer(datastore);

wave_scatter = waveletScattering('SignalLength',8192,'InvarianceScale',0.22,'SamplingFrequency',8000,'OversamplingFactor',2);

rng default;
datastore = shuffle(datastore);
[dataTrain,dataTest] = splitEachLabel(datastore,0.8);

feature_array = cell(numel(dataTrain.Files),1);
%Create Labels for each feature.
%labels = cell(
fprintf("Extracting Features!\n");
for i=1:size(dataTrain.Files)
    [audioIn,fs] = audioread(dataTrain.Files{i});
    aFE = audioFeatureExtractor(...
    "SampleRate",fs, ...
    "Window",hamming(round(0.03*fs),"periodic"), ...
    "OverlapLength",round(0.02*fs), ...
    "mfcc",true, ...
    "mfccDelta",true, ...
    "mfccDeltaDelta",true, ...
    "pitch",true, ...
    "spectralCentroid",true);
    
    feature_array{i} = extract(aFE,audioIn);
    
    fprintf("Done: "+i);
end
fprintf("Training Model");
model = fitcecoc(feature_array,dataTrain.Labels);
