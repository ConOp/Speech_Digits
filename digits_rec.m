recordPath = fullfile('Digit_Dataset','recordings'); %Load the dataset.
datastore = audioDatastore(recordPath);
datastore.Labels = label_distributer(datastore); %Label each audio with the digit it represents.

rng default;
datastore = shuffle(datastore);
[dataTrain,dataTest] = splitEachLabel(datastore,0.8); %split data 80% training the rest 20% for testing.
[silence, fs_silence] = audioread('500-milliseconds-of-silence.mp3');

%Deleting previous testing set.
if isfolder('Test_Dataset')
    files = fullfile('./Test_Dataset','*.wav');
    theFiles = dir(files);
    for k=1:length(theFiles)
       baseFileName = theFiles(k).name;
       fullFileName = fullfile('./Test_Dataset',baseFileName);
       delete(fullFileName);
    end
else
    mkdir('Test_Dataset');
end

%Extract files from testing set.
for i=1:6:numel(dataTest.Files)-6
    [audioIn,fs] = audioread(dataTest.Files{i});
    sample = cell(numel(audioIn),1);
    sample{1} = audioIn;
    for j=1:5
        [sample{j+1},fs] = audioread(dataTest.Files{i+j});
    end    
    %Creating test audio.
    newfile = cat(1,silence,sample{1},silence,sample{2},silence,sample{3},silence,sample{4},silence,sample{5},silence,sample{6},silence);
    n1=double(string(dataTest.Labels(i)));
    n2=double(string(dataTest.Labels(i+1)));
    n3=double(string(dataTest.Labels(i+2)));
    n4=double(string(dataTest.Labels(i+3)));
    n5=double(string(dataTest.Labels(i+4)));
    n6=double(string(dataTest.Labels(i+5)));
    filename = sprintf('%d_%d_%d_%d_%d_%d.wav',n1,n2,n3,n4,n5,n6);   
    fileloc = sprintf('./Test_Dataset/%s',filename);
    if ~isfile(fileloc)        
        audiowrite(fileloc,newfile,fs);
    end
end

feature_array = cell(numel(dataTrain.Files),1);
fprintf("Extracting Features!\n");
for i=1:size(dataTrain.Files)
    [audioIn,fs] = audioread(dataTrain.Files{i});
    %Parameterize Audio Feature Extractor
    aFE = audioFeatureExtractor(...
    "SampleRate",fs, ...
    "Window",hamming(round(0.1*fs),"periodic"), ...
    "OverlapLength",round(0.02*fs), ...
    "mfcc",true, ...
    "mfccDelta",true, ...
    "mfccDeltaDelta",true, ...
    "pitch",true, ...
    "spectralCentroid",true);
    %Extracting features from each audio file in dataset.
    feature_array{i} = extract(aFE,audioIn);
end

sum_rows = 0;

for i=1:size(feature_array)
   [rows, col] = size(feature_array{i});
   sum_rows= sum_rows + rows;
end

labels = zeros(sum_rows,1);
cell_array = vertcat(feature_array{:});
iskip=1;

%Match the features with respectively labels.
for i=1:size(feature_array)
    [rows, col] = size(feature_array{i});
    for j =1:rows
        labels(iskip) = double(string(dataTrain.Labels(i)));
        iskip=iskip+1;
    end
   
end

fprintf("Training Model\n");
model = fitcecoc(cell_array,labels); %Train the model using the features and labels.
saveLearnerForCoder(model, 'newfinalmodel'); % Save the model for later use.

