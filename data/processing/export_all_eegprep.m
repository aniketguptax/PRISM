addpath('/path/to/eeglab');
eeglab;

rootdir = '/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/ds001785/derivatives/eegprep';
outdir  = '/Users/aniketgupta/Desktop/Imperial/FourthYear/FYP/PRISM/data/exports_mat';

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

files = dir(fullfile(rootdir, 'sub-*', 'ses-01', 'eeg', '*_preproc_01hz.set'));
fprintf('Found %d .set files\n', numel(files));

for k = 1:numel(files)
    filepath = fullfile(files(k).folder, files(k).name);
    fprintf('\n[%d/%d] Loading %s\n', k, numel(files), filepath);

    try
        EEG = pop_loadset('filename', files(k).name, 'filepath', [files(k).folder filesep]);

        data = single(EEG.data);   % channels x time x trials
        sfreq = EEG.srate;
        times = EEG.times;
        chanlabels = string({EEG.chanlocs.labels});

        [~, stem, ~] = fileparts(files(k).name);
        outfile = fullfile(outdir, [stem '_export.mat']);

        save(outfile, 'data', 'sfreq', 'times', 'chanlabels', '-v7.3');
        fprintf('Saved %s\n', outfile);

    catch ME
        fprintf('FAILED on %s\n', filepath);
        fprintf('%s\n', ME.message);
    end
end