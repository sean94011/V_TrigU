%% Reset Workspace & Clear Data
clear;
close all;

%% Mode Selection
mode = input("Choose a mode(\n   0 -> Normal Recording, \n   1 -> Recording Profiles Analysis, \n   2 -> RBW Analysis, \n   3 -> Freq BW Analysis, \n   4 -> Freq Points Analysis\n): ");

switch mode
    case 1
        disp('Recording Profiles Analysis...')
    case 2
        disp('RBW Analysis...')
    case 3
        disp('Freq BW Analysis...')
    case 4
        disp('Freq Points Analysis...')
    otherwise
        disp('Normal Recording...')
        normal_recording()
end



