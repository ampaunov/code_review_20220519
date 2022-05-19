#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:14:19 2022

This script runs firstlevel fMRI analyses with parametric modulation 
with a range of options (specified in opt). Specifically:
    
- whether to run temporal denoising at the masking / fmri timecourse 
extraction stage [this has not yet been added to the current version]. 
In addition to the standard motion params, this includes (a) 1st derivatives of 
the motion params, (b) artifact regressors, (c) signals from white matter and 
csf

    
- whether to run the idealObserver with the generative or fitted (on choices)
volatility (note: fitted is currently with fit across all data, behavioral and 
fmri)

- whether to model unmodulated predictors at cue and outcome (+ for questions, 
missed trials)

- whether to model free and forced trials separately

- which modulators to use (from the initializeSubjects dataframe) and where to 
place them: cue, outcome, rt

- which events to model with a duration (and what duration)

*To do*
- not finished: adding the temporal denoising option (include extraction of 
the confound regressors here, so that it's all in one script)

- to move things to input arguments [for eg, either a pre-saved spec of the 
options as path to a json/txt or just each option as an arg + subject range, 
etc]


*Some questions*
how to handle standard folder structure - passing to functions

how to better specify the cross-validation? 


@author: ap267379
"""

import os
import glob
import os.path as op
import json
import pickle
import nibabel as nib

import numpy as np
import pandas as pd
import scipy.stats as sp

from nilearn.input_data import NiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.plotting import plot_design_matrix
from nilearn.glm.contrasts import compute_contrast

import matplotlib.pyplot as plt

from initializeSubject import initialize_subject

pd.options.mode.chained_assignment = None  # default='warn'

# %% opt
opt = {'model_id': 'ERd_EUc_UUm',
       'suffix': 'genvol',
       'descrip': '', 
       'settings': {'denoised': False,  # not included yet (also: to add functions that let you get the confounds if they don't exist? 
                    'fitted_vol': False,
                    'split_free': True, 
                    'split_free_unmod': False,
                    'crossval': True}, 
       'events': {'id': ['cue', 'out', 'q', 'missed'],
                  'duration': [0, 2, 15, 5],
                  'model_unmod': [True, True, True, True]},
       'modulators': {'id': ['ER_diffcu', 'EU_chosen', 'UU_mean', 'RPE'],
                      'place': ['cue', 'cue', 'cue', 'out'],
                      'version': '0'}  #specifies zero-centered, z-scored, raw
       }

# %% select subjects to run
nsub_total = 60
missing_subs = [9, 17, 46]
done_subs = []

subnums = np.arange(1, nsub_total+1)
subnums = list(set(subnums) - set(missing_subs))
subnums = list(set(subnums) - set(done_subs))

# IN ADDITION to dropping the above, specify a range to run 
# (if running several in parallel, for eg)
firstsub = 47  # 1, 21, 41, 51 ||    1, 16, 32, 47
lastsub = 60  # 20, 40, 50, 60 ||   15, 31, 46, 60

subrange = np.arange(firstsub, lastsub+1)

subnums = sorted(list(set(subnums) & set(subrange)))
nsub = len(subnums)

# %% constants
n_trials_block = 96
n_cp = 4
gen_vol = n_cp / n_trials_block
n_blocks_total = 4
if opt['settings']['crossval']:
    n_blocks = n_blocks_total - 1
    n_folds = n_blocks_total
else:
    n_blocks = n_blocks_total
    
n_trials = n_trials_block * n_blocks
n_vols_block = 725
n_vols = n_blocks * n_vols_block
tr = 1.25
block_dur = tr * n_vols_block
block_ons = np.cumsum([0] + [block_dur] * (n_blocks - 1))

oversampling = 50

n_skipped_trials = 4

# %% directories
rootdir = '/home_local/EXPLORE'
datadir = op.join(rootdir, 'SUBJECTS')
niidir = 'nii'
arraysdir = 'fmri_data_arrays'
modeldir = 'firstlevel'
maskdir = op.join(rootdir, 'masks')
maskid = 'mask_Brain_GW_0_2'
maskpath = op.join(maskdir, maskid + '.nii')

# %% Functions

def get_frame_times(TR, n_vols):
    return np.cumsum([TR] * n_vols)

def save_spec(modeldir, opt):
    specid = 'spec'
    existing = glob.glob(op.join(modeldir, specid + '*'))
    n_existing = len(existing)
    specnum = n_existing + 1
    spec_fname = f'{specid}_{specnum:02d}.json'
    specpath = op.join(modeldir, spec_fname)
    fid = open(specpath, 'w')
    json.dump(opt, fid)
    fid.close()

def get_motion_regressors(subject_path, which_blocks=None):
    f_motion = np.asarray(sorted(glob.glob(op.join(subject_path, 'rp_*.txt'))))
    if which_blocks is not None:
        which_blocks = which_blocks - 1 # convert from block nums to indices
        f_motion = f_motion[which_blocks]
    motion = []
    for iblock, f in enumerate(f_motion):
        motion.append(np.loadtxt(f))
    motion = np.vstack(motion)
    n_reg = motion.shape[1]
    reg_names = [f'motion_{param}' for param in np.arange(1, n_reg+1)]
    motion = pd.DataFrame(motion, columns=reg_names)

    return motion

def build_design_matrix(opt, onsets, mods, info, durations, motion=None): 
    dm_cols_ordered = []
    events_unmod = []
    for ievent, event in enumerate(event_cols):
        if opt['events']['model_unmod'][ievent] and np.any(~np.isnan(onsets[event])):
            events_unmod.append(pd.DataFrame({'onset': onsets[event]}))
            events_unmod[ievent]['trial_type'] = event
            if opt['settings']['split_free_unmod']:
                events_unmod[ievent]['trial_type'] = \
                    [event + '_' + freeid for event, freeid in 
                     zip(events_unmod[ievent]['trial_type'], info['free_id'])]
            events_unmod[ievent]['duration'] = durations[ievent]
            events_unmod[ievent]['modulation'] = 1
            dm_cols_ordered.append(np.unique(events_unmod[ievent]['trial_type']))
    
    dm_cols_ordered = np.hstack(dm_cols_ordered)
    events_mod = []
    for imod, mod in enumerate(opt['modulators']['id']):
        mod_place = opt['modulators']['place'][imod]
        events_mod.append(pd.DataFrame({'onset': onsets[mod_place]}))
        events_mod[imod]['trial_type'] = mod_cols[imod] + '_' + mod_place
        if opt['settings']['split_free']:
            events_mod[imod]['trial_type'] = \
                [mod + '_' + freeid for mod, freeid in 
                 zip(events_mod[imod]['trial_type'], info['free_id'])]
        events_mod[imod]['duration'] = durations[np.asarray(opt['events']['id']) == mod_place][0]
        events_mod[imod]['modulation'] = mods[mod_cols[imod]]
        dm_cols_ordered = np.hstack((dm_cols_ordered, np.unique(events_mod[imod]['trial_type'])))
        
    events = pd.concat(events_unmod + events_mod, axis=0, ignore_index=True)
    events = events.loc[~np.isnan(events['onset']), :]
    
    dm = make_first_level_design_matrix(frame_times,
                                        events,
                                        hrf_model='spm',
                                        drift_model=None,
                                        oversampling=oversampling,
                                        add_regs=motion)
    dm.drop(columns=['constant'], inplace=True)
    if motion is not None:
        dm_cols_ordered = np.hstack((dm_cols_ordered, motion.columns))
    
    dm = dm.reindex(columns=dm_cols_ordered)
    # add constants per block
    for iblock in range(n_blocks):
        constant_name = f'constant_{iblock+1}'
        constant = [np.zeros(n_vols_block) for _ in range(n_blocks)]
        constant[iblock] = np.ones(n_vols_block)
        dm[constant_name] = np.hstack(constant)
        
    return dm


def save_design_matrix(dm, opt, subnum, dm_name, print_figure=True):
    subid = f'sub-{subnum:02d}'
    savedir = op.join(datadir, subid, modeldir, opt['model_id'] + '_' + opt['suffix'])
    dm_path = op.join(savedir, dm_name + '.p')
    dm.to_pickle(dm_path)
    if print_figure:
        fig, ax = plt.subplots(figsize=[8, 6])
        plot_design_matrix(dm, ax=ax)
        fig.suptitle(dm_name, y=1.05, fontweight="bold")
        dm_fig_path = op.join(savedir, dm_name + '.png')
        fig.savefig(dm_fig_path, bbox_inches='tight', dpi=220)
        

def initialize_masker(mask_file_path, tr, high_pass=1/128):
    masker = NiftiMasker(mask_img=mask_file_path,
                          standardize=True,
                          high_pass=high_pass,
                          detrend=True,
                          smoothing_fwhm=None,
                          t_r=tr)

    masker.fit()
    return masker


def get_fmri_array(masker, mask_id, subnum, confounds=None, prefix='swtra'):
    subid = f'sub-{subnum:02d}'
    print(f'getting fmri data for {subid}...')
    subdir = op.join(datadir, subid)
    sub_datadir = op.join(subdir, niidir)
    sub_savedir = op.join(subdir, arraysdir)
    nii_files = sorted(glob.glob(op.join(sub_datadir, prefix + '*.nii')))
    array_data = []
    
    if confounds is None:
        suffix = 'nondenoised'
    else:
        suffix = 'denoised'
    
    # confounds are across blocks. Get per block
    # if confounds is not None:
    #     nvols = 725
    #     nblocks = 4
    #     block_idx = np.asarray(
    #         [np.arange(0, nblocks)] * nvols, order='F').ravel(order='F')
    array_exists = 0
    for iblock, nii_file in enumerate(nii_files):
        blocknum = iblock + 1
        array_fname = \
            f'{subid}_{prefix}_{mask_id}_run-{blocknum}_{suffix}.npy'
        array_path = op.join(sub_savedir, array_fname)
        if op.isfile(array_path):
            array_exists += 1
        else:
            if confounds is None:
                array_data.append(masker.transform(nii_file))
            # else:
            #     confounds_block = confounds.loc[block_idx==iblock, :].reset_index()
            #     array_data.append(masker.transform(nii_file, 
            #                                        confounds=confounds_block))
            np.save(array_path, array_data[iblock])
    # print(f'{array_exists}')
    if array_exists == n_blocks_total:
        print('already extracted')
            
def fit_glm(subnum, opt, which_blocks, dm):
    which_blocks = which_blocks - 1 # convert from block numbers to indices
    subid = f'sub-{subnum:02d}'
    sub_datadir = op.join(datadir, subid, arraysdir)
    if opt['settings']['denoised']:
        suffix = 'denoised'
    else:
        suffix = 'nondenoised'
    data_paths = np.asarray(sorted(glob.glob(op.join(sub_datadir, f'*_{suffix}.npy'))))
    data_paths = data_paths[which_blocks]
    fmri_data = []
    for array_path in data_paths:
        fmri_data.append(np.load(array_path))
    fmri_data = np.vstack(fmri_data)
    labels, estimates = run_glm(fmri_data, dm.values)
    
    return labels, estimates


def define_contrasts(opt, dm):
    con_base = pd.DataFrame(np.eye(dm.shape[1]), columns=dm.columns)
    
    mod_ids = opt['modulators']['id']
    mod_places = opt['modulators']['place']
    mod_names = []
    for mod_id, mod_place in zip(mod_ids, mod_places):
        mod_names.append(
            mod_id + '_' + opt['modulators']['version'] + '_' + mod_place)
    
    cons = {}
    for mod_id, mod_name in zip(mod_ids, mod_names):
        if opt['settings']['split_free']:
            mod_id_free = mod_id + '_free'
            mod_name_free = mod_name + '_free'
            mod_id_forced = mod_id + '_forced'
            mod_name_forced = mod_name + '_forced'
            mod_id_diff = mod_id + '_free-forced'
        
            cons[mod_id] = np.mean(con_base[[mod_name_free, mod_name_forced]], axis=1).values
            cons[mod_id_free] = con_base[mod_name_free].values
            cons[mod_id_forced] = con_base[mod_name_forced].values
            cons[mod_id_diff] = np.diff(con_base[[mod_name_forced, mod_name_free]], axis=1).ravel()
        else: 
            cons[mod_id] = con_base[mod_name].values
            
    unmod_ids = [event for event in opt['events']['id'] if event != 'missed']
    for unmod_id in unmod_ids:
        if opt['settings']['split_free_unmod']:
            unmod_id_free = unmod_id + '_free'
            unmod_id_forced = unmod_id + '_forced'
            unmod_id_diff = unmod_id + '_free-forced'
            
            cons[unmod_id] = np.mean(con_base[[unmod_id_free, unmod_id_forced]], axis=1).values
            cons[unmod_id_free] = con_base[unmod_id_free].values
            cons[unmod_id_forced] = con_base[unmod_id_forced].values
            cons[unmod_id_diff] = np.diff(con_base[[unmod_id_forced, unmod_id_free]], axis=1).ravel()
        else:
            cons[unmod_id] = con_base[unmod_id].values
            
    return cons

def estimate_contrasts(labels, estimates, cons, con_type='t'):
    con_estimates = []
    for con in cons:
        con_estimates.append(compute_contrast(labels, estimates, cons[con],
                                              contrast_type=con_type))

    return con_estimates


def save_firstlevel_maps(subnum, con_estimates, cons, masker, opt, 
                         suffix='full', which_maps=['t', 'z', 'con']):
    subid = f'sub-{subnum:02d}'
    savedir = op.join(datadir, subid, modeldir, opt['model_id'] + '_' + opt['suffix'])
    
    for con, con_id in zip(con_estimates, cons):
        con_id = f'{con_id}_{suffix}'
        if 't' in which_maps:
            t_map = masker.inverse_transform(con.stat())
            fname = f'{con_id}_tmap.nii.gz'
            nib.save(t_map, op.join(savedir, fname))
        if 'z' in which_maps:
            z_val = masker.inverse_transform(con.z_score())
            fname = f'{con_id}_zmap.nii.gz'
            nib.save(z_val, op.join(savedir, fname))
        if 'con' in which_maps:

            # Save effect size in a pickle and nii formats
            eff_size_file = f'{con_id}_effect_size.pickle'
            with open(op.join(savedir, eff_size_file), 'wb') as f:
                pickle.dump(con.effect_size(), f)

            effect_size = masker.inverse_transform(con.effect_size())
            eff_size_nii = f'{con_id}_effect_size.nii.gz'
            nib.save(effect_size, op.join(savedir, eff_size_nii))

# %% initialize

mod_v = opt['modulators']['version'] 
if mod_v is not None:
    mod_cols = []
    for mod in opt['modulators']['id']:
        mod_cols.append(f'{mod}_{mod_v}')
else:
    mod_cols = opt['modulators']
    
info_cols = ['block_fmri', 'trial_start', 'rt_start', 'outcome_start', 
             'q_start', 'missed_start', 'free_id', 'rt', 'isfree', 'ismissed', 
             'isQ']

# Full list of events, limited by ones included in opt (e.g., RT not modeled)
event_cols_input = np.asarray(['trial_start', 'rt_start', 'outcome_start', 
                               'q_start', 'missed_start'])
event_cols_all = np.asarray(['cue', 'rt', 'out', 'q', 'missed'])
event_cols_keep = [event_col in opt['events']['id'] for event_col in event_cols_all]
event_cols = event_cols_all[event_cols_keep]
event_cols_input = event_cols_input[event_cols_keep]
n_event_cols = len(event_cols)
durations = np.asarray(opt['events']['duration'])

# get block combinations for glms
if opt['settings']['crossval']:
    blocknums = np.arange(n_blocks_total) + 1
    blocks_used = []
    for block in blocknums:
        blocks_used.append(sorted(list(set(blocknums) - set([block]))))
    blocks_used = np.asarray(blocks_used)
    block_held = blocknums
else:
    blocks_used = np.arange(n_blocks_total) + 1
    block_held = None
    
# set volatilities
if not opt['settings']['fitted_vol']:
    vol = np.asarray([gen_vol] * nsub)
else:
    paramdir = '../Data_neurospin'
    param_path = op.join(paramdir, 'logit_coef_estimates_20220502.xlsx')
    which_params = 'all'
    params = pd.read_excel(param_path, which_params)
    params = params.loc[[sub in subnums for sub in params['subnum'].values]]
    vol = params['vol'].values

# get frame times 
frame_times = get_frame_times(tr, n_vols)

# initialize masker
masker = initialize_masker(maskpath, tr)

# %% Get data: save data arrays (if not already extracted)
for sub in subnums:
    get_fmri_array(masker, maskid, sub, confounds=None, prefix='swtra')

# %% Define design matrix
for isub, sub in enumerate(subnums):
    subid = f'sub-{sub:02d}'
    print(f'running glm for {subid}')
    subdir = op.join(datadir, subid)
    sub_niidir = op.join(subdir, 'nii')
    sub_modeldir = op.join(subdir, 'firstlevel', opt['model_id'] + '_' + opt['suffix'])
    if not op.exists(sub_modeldir):
        os.mkdir(sub_modeldir)
    
    # Save spec file
    save_spec(sub_modeldir, opt)
    
    # Get motion params (if used)
    if opt['settings']['denoised']:
        if opt['settings']['crossval']:
            motion = [None] * n_folds
        else:
            motion = None
    else:
        if opt['settings']['crossval']:
            motion = []
            for ifold in range(n_folds):
                motion.append(
                    get_motion_regressors(sub_niidir, blocks_used[ifold, :]))
        else:
            motion = get_motion_regressors(sub_niidir)
    
    # Get data for design matrix
    data = initialize_subject(sub, vol=vol[isub])
    data = data.loc[data['isfmri'] == 1, :].reset_index()
    
    # Additional info needed [do this in format_behavioral_data?]
    data.loc[:, ['missed_start']] = np.nan
    data.loc[data['ismissed'] == 1, 'missed_start'] = \
        data.loc[data['ismissed'] == 1, 'trial_start']
    # Don't count missed trials as regular trials 
    data.loc[data['ismissed'] == 1, 'trial_start'] = np.nan
    data.loc[data['ismissed'] == 1, 'outcome_start'] = np.nan
    data.loc[:, 'rt_start'] = np.nansum(data[['trial_start', 'rt']], axis=1)
    # Add a column with string ID for free / forced (used in naming)
    data['free_id'] = 'free'
    data.loc[data['isfree'] == 0, 'free_id'] = 'forced'
    data['skipped_trials'] = np.hstack(
        [list(np.ones(n_skipped_trials)) + \
          list(np.zeros(n_trials_block - n_skipped_trials))] * n_blocks_total
            ).astype(bool)
    
    # Split desgin matrix data into folds if crossvalidation used
    if opt['settings']['crossval']:
        data_folds = []
        for ifold in range(n_folds):
            data_folds.append(
                data.loc[data['block_fmri'] != \
                          block_held[ifold], :].reset_index())
    
    # Get modulators and nan-out trials where mods are not modeled
    if opt['settings']['crossval']:
        mods = []
        for ifold in range(n_folds):
            mods.append(data_folds[ifold][mod_cols])
            mods[ifold].loc[data_folds[ifold]['skipped_trials'], :] = np.nan
    else:
        mods = data[mod_cols]
        mods.loc[data['skipped_trials'], :] = np.nan
    
    # Get info
    if opt['settings']['crossval']:
        info = []
        for ifold in range(n_folds):
            info.append(data_folds[ifold][info_cols])
    else:
        info = data[info_cols]

    # Increment onsets across blocks and separate into onsets dataframe    
    if opt['settings']['crossval']:
        onsets = [pd.DataFrame(
            np.nan * np.ones((n_trials, n_event_cols)), columns=event_cols) \
                for _ in range(n_folds)]
        for ifold in range(n_folds):
            for iblock, block in enumerate(blocks_used[ifold, :]):
                for input_col, event_col in zip(event_cols_input, event_cols):
                    onsets[ifold].loc[
                        info[ifold]['block_fmri'] == block, event_col] = \
                        info[ifold].loc[
                            info[ifold]['block_fmri'] == block, input_col] + \
                            block_ons[iblock]
    else:
        onsets = pd.DataFrame(
            np.nan * np.ones((n_trials, n_event_cols)), columns=event_cols)
        for iblock, block in enumerate(np.arange(n_blocks)+1):
            for input_col, event_col in zip(event_cols_input, event_cols):
                onsets.loc[info['block_fmri'] == block, event_col] = \
                    info.loc[info['block_fmri'] == block, input_col] + \
                    block_ons[iblock]
    
    # Get design matrices
    dm_id = subid + '_' + opt['model_id'] + '_' + opt['suffix']
    if opt['settings']['crossval']:
        dm = []
        for ifold in range(n_folds):
            dm.append(build_design_matrix(opt, onsets[ifold], mods[ifold], 
                                          info[ifold], durations, 
                                          motion=motion[ifold]))
            dm[ifold] = pd.DataFrame(sp.zscore(dm[ifold], axis=0), 
                                      columns=dm[ifold].columns, 
                                      index=dm[ifold].index)
            dm_name = f'{dm_id}_crossval_fold-{ifold+1}'
            save_design_matrix(dm[ifold], opt, sub, dm_name, print_figure=True)
    else:
        dm = build_design_matrix(opt, onsets, mods, info, durations, 
                                  motion=motion)
        dm = pd.DataFrame(sp.zscore(dm, axis=0), 
                                    columns=dm.columns, 
                                    index=dm.index)
        save_design_matrix(dm, opt, sub, dm_id, print_figure=True)
    
    # Fit GLM, define contrasts, estimate contrasts, save maps
    if opt['settings']['crossval']:
        labels = []
        estimates = []
        for ifold in range(n_folds):
            fold_id = f'fold-{ifold+1}'
            cons = define_contrasts(opt, dm[ifold])
            labels, estimates = \
                fit_glm(sub, opt, blocks_used[ifold, :], dm[ifold])
            con_est = estimate_contrasts(labels, estimates, cons, con_type='t')
            save_firstlevel_maps(sub, con_est, cons, masker, opt, 
                          suffix=fold_id, which_maps=['t', 'z', 'con'])
            
    else: 
        cons = define_contrasts(opt, dm)
        labels, estimates = fit_glm(sub, opt, blocks_used, dm)
        con_est = estimate_contrasts(labels, estimates, cons, con_type='t')
        save_firstlevel_maps(sub, con_est, cons, masker, opt, 
                          suffix='full', which_maps=['t', 'z', 'con'])
            