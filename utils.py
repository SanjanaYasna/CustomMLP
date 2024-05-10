import pandas as pd
import numpy as np
import joblib
import os
# scale features
from sklearn import preprocessing
from sklearn import impute
# classifier
from sklearn.ensemble import ExtraTreesClassifier
# scoring metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
#get colors
import color as color
class Utils:
    #takes in a (test) dataframe and returns a dataframe with the same columns but with:
    # 1. All columns that are not in the feature set removed
    # 2. Extraneous columns  and bad terms emoved
    def feature_subset(df, subset, other_bad_terms = [], noBSA=False):
        X = df.copy()
        X = X.fillna(0)
        not_needed = ("Catalytic", "SITE_ID", "Set", "ValidSet", 'NewSet', 'cath_class', 'cath_arch', 'scop_class', 'scop_fold', 'ECOD_arch', 'ECOD_x_poshom', 'ECOD_hom')
        X = X.drop(columns = [term for term in X if term.startswith(not_needed)])
        bad_terms = ("hbond_lr_", 'dslf_fa13', 'pro_close', 'ref', 'fa_sol_', 'MetalCodes', 'MetalAtoms', 'SEPocket', 'geom_gRMSD', 'geom_MaxgRMSDDev', 'geom_AtomRMSD')
        X = X.drop(columns = [term for term in X if term.startswith(bad_terms)])
        #print(X.shape)#, list(X))
        #general terms
        gen_set = ['Depth', 'Vol', "SITEDistCenter", "SITEDistNormCenter"]
        gen_terms = ("BSA", 'SASA')
        all_gen_set = [ term for term in X if term.startswith(gen_terms) ]
        gen_shell = [name for name in all_gen_set if "_S" in name]
        gen_sph = list(set(all_gen_set).difference(gen_shell))
        gen_shell += gen_set
        gen_shell += ["BSA_3.5", "SASA_3.5"]
        gen_sph += gen_set
        all_gen_set += gen_set
        all_gen_set = sorted(set(all_gen_set))
        #Rosetta terms only
        ros_sum_sph0 = list(set([name for name in X if name.endswith("_Sum_3.5")]).difference(all_gen_set))
        ros_sum_sph1 = list(set([ name for name in X if name.endswith("_Sum_5") ]).difference(all_gen_set))
        ros_sum_sph2 = list(set([ name for name in X if name.endswith("_Sum_7.5") ]).difference(all_gen_set))
        ros_sum_sph3 = list(set([ name for name in X if name.endswith("_Sum_9") ]).difference(all_gen_set))
        ros_sum_shell1 = list(set([ name for name in X if name.endswith("_Sum_S5") ]).difference(all_gen_set))
        ros_sum_shell2 = list(set([ name for name in X if name.endswith("_Sum_S7.5") ]).difference(all_gen_set))
        ros_sum_shell3 = list(set([ name for name in X if name.endswith("_Sum_S9") ]).difference(all_gen_set))
        ros_sum_shell = ros_sum_sph0 + ros_sum_shell1 + ros_sum_shell2 + ros_sum_shell3
        ros_sum_sph = ros_sum_sph0 + ros_sum_sph1 + ros_sum_sph2 + ros_sum_sph3

        ros_mean_sph0 = list(set([name for name in X if name.endswith("_Mean_3.5")]).difference(all_gen_set))
        ros_mean_sph1 = list(set([ name for name in X if name.endswith("_Mean_5") ]).difference(all_gen_set))
        ros_mean_sph2 = list(set([ name for name in X if name.endswith("_Mean_7.5") ]).difference(all_gen_set))
        ros_mean_sph3 = list(set([ name for name in X if name.endswith("_Mean_9") ]).difference(all_gen_set))
        ros_mean_shell1 = list(set([ name for name in X if name.endswith("_Mean_S5") ]).difference(all_gen_set))
        ros_mean_shell2 = list(set([ name for name in X if name.endswith("_Mean_S7.5") ]).difference(all_gen_set))
        ros_mean_shell3 = list(set([ name for name in X if name.endswith("_Mean_S9") ]).difference(all_gen_set))
        ros_mean_shell = ros_mean_sph0 + ros_mean_shell1 + ros_mean_shell2 + ros_mean_shell3
        ros_mean_sph = ros_mean_sph0 + ros_mean_sph1 + ros_mean_sph2 + ros_mean_sph3

        electro = [name for name in X if name.startswith("Elec")]
        geom = [name for name in X if name.startswith("geom")]
        findgeo_geoms = ("lin", "trv", "tri", "tev", "spv",
            "tet", "spl", "bva", "bvp", "pyv",
            "spy", "tbp", "tpv",
            "oct", "tpr", "pva", "pvp", "cof", "con", "ctf", "ctn",
            "pbp", "coc", "ctp", "hva", "hvp", "cuv", "sav",
            "hbp", "cub", "sqa", "boc", "bts", "btt",
            "ttp", "csa")
        geom = [name for name in geom if not name.endswith(findgeo_geoms)] #remove the individual geom types
        #pocket features only
        pocket_set = ['Depth', 'Vol', "SITEDistCenter", "SITEDistNormCenter", 'LongPath', 'farPtLow', 'PocketAreaLow', 'OffsetLow', 'LongAxLow', 'ShortAxLow', 'farPtMid', 'PocketAreaMid', 'OffsetMid', 'LongAxMid', 'ShortAxMid', 'farPtHigh', 'PocketAreaHigh', 'OffsetHigh', 'LongAxHigh', 'ShortAxHigh']
        pocket_set = list(set(pocket_set).difference(other_bad_terms))
        #pocket lining only
        lining_set = ['num_pocket_bb', 'num_pocket_sc', 'avg_eisen_hp', 'min_eisen', 'max_eisen', 'skew_eisen', 'std_dev_eisen', 'avg_kyte_hp', 'min_kyte', 'max_kyte', 'skew_kyte', 'std_dev_kyte', 'occ_vol', 'NoSC_vol', 'SC_vol_perc', 'LiningArea']
        lining_set = list(set(lining_set).difference(other_bad_terms))

        #print( len(all_gen_set), len(sorted(set(ros_sum_sph+ros_sum_shell+ros_mean_shell+ros_mean_sph))), len(electro), len(geom), len(pocket_set), len(lining_set))
        #print(len(sorted(set(ros_sum_sph+ros_sum_shell+ros_mean_shell+ros_mean_sph+all_gen_set+electro+geom+pocket_set+lining_set))))

        subset_list = ["AllSumSph", "AllMeanSph", "AllSumShell", "AllMeanShell",
                        "GenSph", "GenShell", "Pocket", "Lining",
                        'RosSumSph', 'RosSumSph0', 'RosSumSph1', 'RosMeanSph', 'RosMeanSph0', 'RosMeanSph1', "RosSumSphInner2", "RosMeanSphInner2",
                        'RosSumShell', 'RosSumShell1', 'RosMeanShell', 'RosMeanShell1',"RosSumShellInner2", "RosMeanShellInner2",
                        "LinPocket", "LinRosSumSph", "LinRosMeanSph", "LinRosSumShell", "LinRosMeanShell",
                        "PocketRosSumSph", "PocketRosMeanSph", "PocketRosSumShell", "PocketRosMeanShell",
                        "Geom", "LinPocketGeom", "GeomElectro", "GeomRosSumSph", "GeomRosSumShell", "GeomRosMeanSph", "GeomRosMeanShell",

                        "Electro", "LinPocketElectro", "LinPocketElectroGeom", "ElectroRosSumSph", "ElectroRosSumShell", "ElectroRosMeanSph", "ElectroRosMeanShell",
                        "AllSumSphMinusGen", "AllSumSphMinusLin", "AllSumSphMinusPocket",
                        "AllSumSphMinusGeom", "AllSumSphMinusElectro",
                        "AllMeanSphMinusGen", "AllMeanSphMinusLin", "AllMeanSphMinusPocket",
                        "AllMeanSphMinusGeom", "AllMeanSphMinusElectro", "AllMinusRosSph",

                        "AllSumShellMinusGen", "AllSumShellMinusLin", "AllSumShellMinusPocket",
                        "AllSumShellMinusGeom", "AllSumShellMinusElectro",
                        "AllMeanShellMinusGen", "AllMeanShellMinusLin", "AllMeanShellMinusPocket",
                        "AllMeanShellMinusGeom", "AllMeanShellMinusElectro", "AllMinusRosShell",
                        ]
        column_subsets = [  sorted(set(gen_sph+ros_sum_sph+pocket_set+lining_set+electro+geom)),#AllSumSph  GSP
                            sorted(set(gen_shell+ros_mean_sph+pocket_set+lining_set+electro+geom)),#AllMeanSph  GSP
                            sorted(set(gen_sph+ros_sum_shell+pocket_set+lining_set+electro+geom)),#AllSumShell  GSH
                            sorted(set(gen_shell+ros_mean_shell+pocket_set+lining_set+electro+geom)), #AllMeanShell GSH
                            gen_sph,#GenSph GSH
                            gen_shell,#GenShell GPH

                            pocket_set,#Pocket
                            lining_set,#Lining
                            ros_sum_sph,#RosSumSph
                            ros_sum_sph0,#RosSumSph0
                            ros_sum_sph1,#RosSumSph1
                            ros_mean_sph,#RosMeanSph
                            ros_mean_sph0,#RosMeanSph0
                            ros_mean_sph1,#RosMeanSph1
                            sorted(set(ros_sum_sph0+ros_sum_sph1)),#RosSumSphInner2
                            sorted(set(ros_mean_sph0+ros_mean_sph1)),#RosMeanSphInner2
                            ros_sum_shell, #RosSumShell
                            ros_sum_shell1, #RosSumShell1
                            ros_mean_shell, #RosMeanShell
                            ros_mean_shell1, #RosMeanShell1
                            sorted(set(ros_sum_sph0 + ros_sum_shell1)), #RosSumShellInner2
                            sorted(set(ros_mean_sph0 + ros_mean_shell1)), #RosMeanShellInner2
                            lining_set+pocket_set, #LinPocket
                            lining_set+ros_sum_sph, #LinRosSumSph
                            lining_set+ros_mean_sph, #LinRosMeanSph
                            lining_set+ros_sum_shell, #LinRosSumShell
                            lining_set+ros_mean_shell, #LinRosMeanShell
                            pocket_set+ros_sum_sph, #PocketRosSumSph
                            pocket_set+ros_mean_sph, #PocketRosMeanSph
                            pocket_set+ros_sum_shell, #PocketRosSumShell
                            pocket_set+ros_mean_shell, #PocketRosMeanShell
                            geom, #Geom
                            lining_set+pocket_set+geom, #LinPocketGeom
                            geom+electro, #GeomElectro
                            geom+ros_sum_sph, #GeomRosSumSph
                            geom+ros_sum_shell,#GeomRosSumShell
                            geom+ros_mean_sph, #GeomRosMeanSph
                            geom+ros_mean_shell,#GeomRosMeanShell

                            electro,#Electro
                            lining_set+pocket_set+electro,#LinPocketElectro
                            lining_set+pocket_set+electro+geom,#LinPocketElectroGeom
                            electro+ros_sum_sph,#ElectroRosSumSph
                            electro+ros_sum_shell,#ElectroRosSumShell
                            electro+ros_mean_sph,#ElectroRosMeanSph
                            electro+ros_mean_shell,#ElectroRosMeanShell
                            sorted(set(ros_sum_sph+pocket_set+lining_set+electro+geom)),#AllSumSphMinusGen
                            sorted(set(gen_sph+ros_sum_sph+pocket_set+electro+geom)),#AllSumSphMinusLin   GSP
                            sorted(set(gen_sph+ros_sum_sph+lining_set+electro+geom)),#AllSumSphMinusPocket  GSP
                            sorted(set(gen_sph+ros_sum_sph+pocket_set+lining_set+electro)),#AllSumSphMinusGeom  GSP
                            sorted(set(gen_sph+ros_sum_sph+pocket_set+lining_set+geom)), #AllSumSphMinusElectro  GSP
                            sorted(set(ros_mean_sph+pocket_set+lining_set+electro+geom)),#AllMeanSphMinusGen
                            sorted(set(gen_sph+ros_mean_sph+pocket_set+electro+geom)),#AllMeanSphMinusLin   GSP
                            sorted(set(gen_sph+ros_mean_sph+lining_set+electro+geom)),#AllMeanSphMinusPocket  GSP
                            sorted(set(gen_sph+ros_mean_sph+pocket_set+lining_set+electro)),#AllMeanSphMinusGeom  GSP
                            sorted(set(gen_sph+ros_mean_sph+pocket_set+lining_set+geom)),#AllMeanSphMinusElectro  GSP
                            sorted(set(gen_sph+pocket_set+lining_set+electro+geom)),#AllMinusRosSph  GSP

                            sorted(set(ros_sum_shell+pocket_set+lining_set+electro+geom)),#AllSumShellMinusGen
                            sorted(set(gen_shell+ros_sum_shell+pocket_set+electro+geom)),#AllSumShellMinusLin  GSH
                            sorted(set(gen_shell+ros_sum_shell+lining_set+electro+geom)),#AllSumShellMinusPocket  GSH
                            sorted(set(gen_shell+ros_sum_shell+pocket_set+lining_set+electro)),#AllSumShellMinusGeom  GSH
                            sorted(set(gen_shell+ros_sum_shell+pocket_set+lining_set+geom)), #AllSumShellMinusElectro  GSH
                            sorted(set(ros_mean_shell+pocket_set+lining_set+electro+geom)),#AllMeanShellMinusGen
                            sorted(set(gen_shell+ros_mean_shell+pocket_set+electro+geom)),#AllMeanShellMinusLin  GSH
                            sorted(set(gen_shell+ros_mean_shell+lining_set+electro+geom)),#AllMeanShellMinusPocket  GSH
                            sorted(set(gen_shell+ros_mean_shell+pocket_set+lining_set+electro)), #AllMeanShellMinusGeom  GSH
                            sorted(set(gen_shell+ros_mean_shell+pocket_set+lining_set+geom)),#AllMeanShellMinusElectro   GSH
                            sorted(set(gen_shell+pocket_set+lining_set+electro+geom)),#AllMinusRosShell  GSH
                            ]
        #print(column_subsets[subset_list.index(data_subset)] )
        if subset in subset_list:
            X = X[ column_subsets[subset_list.index(subset)] ]
        else:
            print("Not a subset in list; defaulting to AllSph")
            X = X[ column_subsets[0] ] #this is all for usage with PCA/UMAP; it uses the rosetta sphere terms plus all the non-rosetta terms
        if 'groupID' in df.columns:
            X=pd.merge(X,df['groupID'], left_index=True, right_index=True)
        ## added to remove BSA terms for undersampling DataSet using BSA
        if noBSA==True:
            X = X.drop(columns = [term for term in X if term.startswith("BSA")])
            X = X.drop(columns = [term for term in X if term.startswith("SASA")])
        return(X)
    
    
    def get_scaled_features(sites, pkl_out, save_models = None):
        #change to pkl_out folder directory
        if pkl_out is not None:
            os.chdir(pkl_out)
        # seperate the sets (only dataset will be used to set scaling)
        data = sites.loc[sites.Set == "data"].copy()
        Tsites = sites.loc[sites.Set == "test"].copy()

        #split for scaling into categorical and not categorical
        not_ctg_geom = ("geom_gRMSD", "geom_MaxgRMSDDev","geom_val", "geom_nVESCUM","geom_AtomRMSD", "geom_AvgO", "geom_AvgN", "geom_AvgS", "geom_AvgOther", "geom_Charge")
        geom = [name for name in data if name.startswith("geom")]

        ctg_data = [x for x in geom if not x in not_ctg_geom]
        ctg_data.extend(["Set", 'Catalytic'])

        ## scale cont. features
        cont_scaler = preprocessing.RobustScaler(quantile_range=(20,80))
        #Fit scaler to X, then transform it
        data_nonctg = data[data.columns.difference(ctg_data)]#so that I can have columns
        data_scaled = pd.DataFrame(cont_scaler.fit_transform(data_nonctg), columns=data_nonctg.columns, index=data_nonctg.index)

        #scale the test set based on the scale of the training set
        Tsites_nonctg  = Tsites[Tsites.columns.difference(ctg_data)]
        Tsites_scaled = pd.DataFrame(cont_scaler.transform(Tsites_nonctg), columns=Tsites_nonctg.columns, index=Tsites_nonctg.index)

        #replace continuous feature null values with mean
        cont_imputer = impute.SimpleImputer(strategy="mean")
        data_scaled = pd.DataFrame(cont_imputer.fit_transform(data_scaled), columns=data_scaled.columns, index=data_scaled.index)
        Tsites_scaled = pd.DataFrame(cont_imputer.transform(Tsites_scaled), columns=Tsites_scaled.columns, index=Tsites_scaled.index)

        if save_models==True:
            joblib.dump(cont_scaler, "/ContVarScaler.pkl")
            joblib.dump(cont_imputer, "/ContVarImpute.pkl")

        #remove groupID and target value Catalytic so that it also isn't MinMax scaled either
        ctg_data.remove("Set");ctg_data.remove("Catalytic");
        #transform categorical data to [0,1] interval using fit_transform (StandardScaler), and then imputer for null values
        if len(data.columns.intersection(ctg_data)) > 0:
            ctg_scaler = preprocessing.MinMaxScaler()

            # fit the scaler to the data-set (training) and scale
            data_ctg = data[data.columns.intersection(ctg_data)]
            data_ctg_scaled = pd.DataFrame(ctg_scaler.fit_transform(data_ctg), columns=data_ctg.columns, index=data_ctg.index)

            #scale the test set based on the scale of the training set
            Tsites_ctg = Tsites[Tsites.columns.intersection(ctg_data)]
            Tsites_ctg_scaled = pd.DataFrame(ctg_scaler.transform(Tsites_ctg), columns=Tsites_ctg.columns, index=Tsites_ctg.index)

            #replace categoric features null values with median value
            ctg_imputer = impute.SimpleImputer(strategy="median")
            data_ctg_scaled = pd.DataFrame(ctg_imputer.fit_transform(data_ctg_scaled), columns=data_ctg_scaled.columns, index=data_ctg_scaled.index)
            Tsites_ctg_scaled = pd.DataFrame(ctg_imputer.transform(Tsites_ctg_scaled), columns=Tsites_ctg_scaled.columns, index=Tsites_ctg_scaled.index)

            #concatenate the scaled categorical data to the robustly scaled data
            data_scaled = pd.merge(data_scaled, data_ctg_scaled, left_index=True, right_index=True)
            Tsites_scaled = pd.merge(Tsites_scaled, Tsites_ctg_scaled, left_index=True, right_index=True)

            if save_models==True:
                joblib.dump(ctg_scaler, "/CtgVarScaler.pkl")
                joblib.dump(ctg_imputer, "/CtgVarImpute.pkl")
        #add back the Catalytic column
        data_scaled = pd.merge(data_scaled, data['Catalytic'], left_index=True, right_index=True)
        Tsites_scaled = pd.merge(Tsites_scaled, Tsites['Catalytic'], left_index=True, right_index=True)

        return(data_scaled, Tsites_scaled)

    ##returns relevent data-set data for training ML model
    def get_training_data(feature_set, random_seed, data_scaled):
        ## random under sample data-set (1+:3-)
        X_Cat = data_scaled[data_scaled['Catalytic']==True].copy()
        X_nonCat = data_scaled[data_scaled['Catalytic']==False].copy()
        #sample X_nonCat so that it is 3 parts non-Catalytic to 1 part Catalytic data
        #NOTE: allow reuse of samples (duplicate sites, possibly) to avoid issue of sample size errors if the catalytic site is less than 3 times the non catalytic site count.
        #Make sure to toggle replace to false if preferred
        X_nonCat = X_nonCat.sample(n=len(X_Cat)*3, axis=0, random_state=random_seed, replace = True)
        X_prep = pd.concat([X_Cat, X_nonCat], axis = 0)
    # X_Cat.append(X_nonCat)

        ## seperate target value
        y = X_prep['Catalytic']; del X_prep['Catalytic']

        ## only return features in specific feature set
        X = Utils.feature_subset(X_prep, feature_set, noBSA=True)

        return(X, y)

    ## number of iterations to improve reproducability
    def evaluate_model_with_Tsite(clf, feature_set, Tsites_scaled, save_models=False, num_rand_seeds=10):
        ## prepare test-set
        testX = Tsites_scaled.copy()
        testY = testX['Catalytic']; del testX['Catalytic']
        testX = Utils.feature_subset(testX, feature_set, noBSA=True)

        ## get multiple predictions for test-set w/ diff random seeds
        test_site_preds = {'actual': pd.Series(testY, index=testX.index)}
        for rand_seed in range(0,num_rand_seeds):
            # get undersampled training data for feature set
            X, y = Utils.get_training_data(feature_set = feature_set, random_seed=rand_seed, data_scaled=Tsites_scaled)
            print("random_seed = %s"%(rand_seed), end="\t")
            print("(num. training sites= %s (%s+ : %s-) \tnum. features: %s)"%(X.shape[0], len(y[y==True]),len(y[y==False]), X.shape[1]))

            ## train classifier and make test-set predictions (alreacy put in random seed when doing hte relevant data sampling and splitting in the @get_training_data)
            #The model itself has a random state of 0 - 9 for each of htese iterations
            clf.set_params(random_state=rand_seed)
            #fit in training set with random seed
            clf.fit(X, y)
            #predict with test-set input
            test_preds = clf.predict(testX)
            test_site_preds['prediction_%s'%(rand_seed)]= pd.Series(test_preds, index=testX.index)
            if save_models==True:
                joblib.dump(clf, "/MAHOMES%s.pkl"%(rand_seed))

            ## output results for this random seed to get an idea of prediction variation levels
            TN, FP, FN, TP = confusion_matrix(testY, test_preds).ravel()
            mcc = matthews_corrcoef(testY, test_preds)
            print("\tTP=%s \tTN=%s \tFP=%s \tFN=%s"%(TP, TN, FP, FN))

        ## calcualte the average of all random seed predictions
        test_predictions = pd.DataFrame(test_site_preds)
        test_predictions['prediction']=0
        for rand_seed in range(0,num_rand_seeds):
            test_predictions['prediction']+=test_predictions['prediction_%s'%(rand_seed)]
        test_predictions['prediction']=test_predictions['prediction']/num_rand_seeds

        ## make final prediction
        test_predictions['bool_pred']=False
        test_predictions.loc[test_predictions['prediction']>=0.5, 'bool_pred']=True

        return(test_predictions)

    ## return result metrics for final predictions
    def check_result_metrics(alg, feat_set, prediction_df):
        mcc = matthews_corrcoef(prediction_df['actual'], prediction_df['bool_pred'])
        TN, FP, FN, TP = confusion_matrix(prediction_df['actual'], prediction_df['bool_pred']).ravel()

        TPR=(TP/(TP+FN))*100
        TNR=(TN/(TN+FP))*100
        acc=((TP+TN)/(TP+TN+FP+FN))*100
        Prec=(TP/(TP+FP))*100
        return(pd.DataFrame([[alg, feat_set, acc, mcc, TPR, TNR, Prec]],
            columns=['Algorithm', 'Feature Set', 'Accuracy', 'MCC', 'Recall', 'TrueNegRate', 'Precision']))
    #keras version of the above:
    def check_result_metrics_for_keras(alg, feat_set, prediction_df):
        mcc = matthews_corrcoef(prediction_df['actual'], prediction_df['bool_pred'])
        TN, FP, FN, TP = confusion_matrix(prediction_df['actual'], prediction_df['bool_pred']).ravel()

        TPR=(TP/(TP+FN))*100
        TNR=(TN/(TN+FP))*100
        acc=((TP+TN)/(TP+TN+FP+FN))*100
        Prec=(TP/(TP+FP))*100
        return(pd.DataFrame([[alg, feat_set, acc, mcc, TPR, TNR, Prec]],
            columns=['Algorithm', 'Feature Set', 'Accuracy', 'MCC', 'Recall', 'TrueNegRate', 'Precision']))