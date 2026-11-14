#!/usr/bin/env python3
"""Five-fold trainval evaluation matching teammate baseline hyperparameters.
Outputs fold metrics and OOF predictions; does not access held-out test data.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, confusion_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('.'))
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p.add_argument('--protocol', choices=['isotropic_1mm', 'original_spacing'], default='isotropic_1mm')
    args = p.parse_args()
    splits = pd.read_csv(args.data_dir / 'splits.csv')
    nodules = pd.read_csv(args.data_dir / 'nodules.csv')
    features = pd.read_csv(args.data_dir / 'features_trainval.csv')
    for name, frame, cols in [('splits',splits,['voi_name','patient_id','split','fold']),('nodules',nodules,['voi_name','diameter_mm','malignant']),('features',features,['voi_name','patient_id','malignant','fold','protocol'])]:
        missing = set(cols)-set(frame.columns)
        if missing: raise ValueError(f'{name} missing columns: {missing}')
    trainval = splits.loc[splits['split']=='trainval'].copy()
    if len(trainval)!=828 or set(trainval.fold.unique())!=set(range(5)):
        raise ValueError('Unexpected trainval size or fold IDs; inspect input files')
    if trainval.voi_name.duplicated().any() or nodules.voi_name.duplicated().any():
        raise ValueError('Duplicate nodule keys')
    if trainval.groupby('patient_id').fold.nunique().max()!=1:
        raise ValueError('Patient leakage across folds')
    f = features.loc[features.protocol==args.protocol].copy()
    if f.voi_name.duplicated().any(): raise ValueError('Duplicate radiomics keys')
    df = trainval.merge(nodules[['voi_name','diameter_mm','malignant']], on='voi_name',validate='one_to_one')
    df = df.merge(f.drop(columns=['patient_id','malignant','fold','protocol']),on='voi_name',validate='one_to_one')
    if len(df)!=len(trainval): raise ValueError('Incomplete feature join')
    check = trainval.merge(f[['voi_name','patient_id','malignant','fold']],on='voi_name',validate='one_to_one')
    if len(check)!=len(trainval) or not (check.patient_id_x==check.patient_id_y).all() or not (check.fold_x==check.fold_y).all():
        raise ValueError('Feature/split patient or fold mismatch')
    labels = trainval.merge(nodules[['voi_name','malignant']],on='voi_name').merge(f[['voi_name','malignant']],on='voi_name',suffixes=('_nodule','_feature'))
    if not (labels.malignant_nodule==labels.malignant_feature).all(): raise ValueError('Label mismatch')
    if not set(df.malignant.unique()).issubset({0,1}): raise ValueError('Labels must be 0/1')
    radiomics = [c for c in f.columns if c.startswith('original_') and c not in {'original_spacing_x','original_spacing_y','original_spacing_z'}]
    if not radiomics: raise ValueError('No radiomics feature columns found')
    for c in ['diameter_mm']+radiomics:
        df[c] = pd.to_numeric(df[c],errors='raise')
    if not np.isfinite(df[['diameter_mm']+radiomics].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite feature values; inspect input data')
    feature_sets = {'Diameter':['diameter_mm'],'Radiomics':radiomics,'Diameter + Radiomics':['diameter_mm']+radiomics}
    rows, preds = [], []
    for method, cols in feature_sets.items():
        for fold in range(5):
            tr = df.fold != fold
            va = df.fold == fold
            xtr, xva = df.loc[tr,cols], df.loc[va,cols]
            ytr, yva = df.loc[tr,'malignant'].astype(int),df.loc[va,'malignant'].astype(int)
            if ytr.nunique()!=2 or yva.nunique()!=2: raise ValueError(f'Single-class fold {fold}')
            model = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('classifier',LogisticRegression(C=1.0,max_iter=5000,solver='lbfgs'))])
            model.fit(xtr,ytr)
            score = model.predict_proba(xva)[:,1]
            pred = (score>=0.5).astype(int)
            tn,fp,fn,tp=confusion_matrix(yva,pred,labels=[0,1]).ravel()
            rows.append({'method':method,'protocol':args.protocol,'fold':fold,'n_train':int(tr.sum()),'n_validation':int(va.sum()),'n_features':len(cols),'auc':roc_auc_score(yva,score),'average_precision':average_precision_score(yva,score),'accuracy':accuracy_score(yva,pred),'balanced_accuracy':balanced_accuracy_score(yva,pred),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp)})
            preds.extend({'method':method,'protocol':args.protocol,'fold':fold,'voi_name':name,'y_true':int(y),'y_score':float(s),'y_pred':int(pr)} for name,y,s,pr in zip(df.loc[va,'voi_name'],yva,score,pred))
    fold_results=pd.DataFrame(rows)
    summary=[]
    for method,group in fold_results.groupby('method',sort=False):
        entry={'method':method,'protocol':args.protocol,'n_trainval':len(df),'n_features':int(group.n_features.iloc[0])}
        for metric in ['auc','average_precision','accuracy','balanced_accuracy','sensitivity','specificity']:
            entry[f'{metric}_mean']=group[metric].mean()
            entry[f'{metric}_std']=group[metric].std(ddof=1)
        pr=pd.DataFrame([r for r in preds if r['method']==method])
        entry['oof_auc']=roc_auc_score(pr.y_true,pr.y_score)
        entry['oof_average_precision']=average_precision_score(pr.y_true,pr.y_score)
        summary.append(entry)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fold_results.to_csv(args.output_dir/'results_by_fold.csv',index=False,float_format='%.6f')
    pd.DataFrame(summary).to_csv(args.output_dir/'results_summary.csv',index=False,float_format='%.6f')
    pd.DataFrame(preds).to_csv(args.output_dir/'oof_predictions.csv',index=False,float_format='%.6f')
    print(f'Validated {len(df)} trainval nodules; {len(radiomics)} radiomics features; protocol={args.protocol}')
    print(pd.DataFrame(summary)[['method','auc_mean','auc_std','accuracy_mean','oof_auc']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(f'Outputs saved in {args.output_dir.resolve()}')

if __name__=='__main__': main()
