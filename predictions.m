function digits=predictions(filename)

    model = loadLearnerForCoder('BestAttempt');%Load previously trained model.
    [y,fs]=audioread(filename); %Load the audio for recognition.
    envelope = imdilate(abs(y), true(1500, 1));
    quietParts = abs(envelope) > 0.01; %Threshhold to detect speech.
    beginning = strfind(quietParts',[0 1]);
    ending = strfind(quietParts', [1 0]);

    parts = cell(numel(ending),1);
    for i=1:numel(ending) 
        parts{i}=y(beginning(i):ending(i)); %Separate detected words into parts.
    end
    results = zeros(numel(parts),1);
    %Extract features for each part.
    for i=1:numel(parts)
        aFE = audioFeatureExtractor(...
        "SampleRate",fs, ...
        "Window",hamming(round(0.1*fs),"periodic"), ...
        "OverlapLength",round(0.02*fs), ...
        "mfcc",true, ...
        "mfccDelta",true, ...
        "mfccDeltaDelta",true, ...
        "pitch",true, ...
        "spectralCentroid",true);

        feature_array = extract(aFE,parts{i});
        %Predict the digit from part.
        [~, score] = predict(model,feature_array);
        [m,i1] = max(score,[],2);

        [~,i2] = max(m);
        results(i) = i1(i2)-1;
        
    end
    %Function returns the results.
    digits=results;
end

