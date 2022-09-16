%% Reset Workspace & Clear Data
clear;
close all;

%% Load Library
% add DOTNET assembly
NET.addAssembly([getenv('programfiles'),'\Vayyar\vtrigU\bin\vtrigU.CSharp.dll']);

% Load Module
import vtrigU.*;

% add plotting functions
addpath('plot\')

%%