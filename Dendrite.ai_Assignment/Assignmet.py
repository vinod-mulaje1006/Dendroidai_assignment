import json
import pandas as pd 

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.neighbors import KNeighborsClassifier

# Regressors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split


# Specify the file path
file_path = 'algoparams.json'

# Open and read the file
jsonString=''
with open(file_path, 'r') as file:
    config = json.load(file)

# f = open(file_path, 'r')
# for x in f:
#   jsonString +=x
# config = json.loads(jsonString)

csv_path = config["design_state_data"]["session_info"]["dataset"]

data = pd.read_csv(csv_path)
print(data)



# for feature_name, feature_details in config["design_state_data"]["feature_handling"].items():
#     print('*'*5,feature_name,'*'*5)
#     for key, value in feature_details.items():  # Iterate over the keys and values
#         print(f"{key}: {value}")  # Print the key and its corresponding value
#     print()
    

for feature_name, feature_details in config["design_state_data"]["feature_handling"].items():
    if feature_name in data.columns:
        missing_values_action = feature_details.get("feature_details", {}).get("missing_values", "")
        print(missing_values_action)
        impute_with = feature_details.get("feature_details", {}).get("impute_with", "")
        impute_value = feature_details.get("feature_details", {}).get("impute_value", 0)
        print('impute_with ', impute_with)
        print('impute_value ', impute_value)

        if missing_values_action == "Impute":
            if impute_with == "Average of values":
                data[feature_name].fillna(data[feature_name].mean(), inplace=True)
            elif impute_with == "Median of values":
                data[feature_name].fillna(data[feature_name].median(), inplace=True)
            elif impute_with == "custom":
                data[feature_name].fillna(impute_value, inplace=True)
            else:
                data[feature_name].fillna(data[feature_name].mean(), inplace=True)



target_column = config["design_state_data"]["target"]["target"]
print('target_column ', target_column)
target = data[target_column]

# Extract feature reduction configuration
reduction_config = config["design_state_data"]["feature_reduction"]
reduction_method = reduction_config["feature_reduction_method"]
num_features_to_keep = int(reduction_config.get("num_of_features_to_keep", len(data.columns)))
print('reduction_config ', reduction_config)
print('reduction_method ', reduction_method)
print('num_features_to_keep ', num_features_to_keep)

# Apply feature reduction based on the method specified
if reduction_method == "No Reduction":
    print("\nNo feature reduction applied.")
    reduced_data = data

elif reduction_method == "Corr with Target":
    print("\nApplying Correlation with Target for feature reduction.")
    correlations = data.corrwith(target)
    top_features = correlations.abs().sort_values(ascending=False).head(num_features_to_keep).index
    print(f"Selected features based on correlation: {list(top_features)}")
    reduced_data = data[top_features]

elif reduction_method == "Tree-based":
    print("\nApplying Tree-based feature reduction.")
    categorical_columns = data.select_dtypes(include=['object']).columns
    print('Categorical columns:', categorical_columns)
    if not categorical_columns.empty:
        print(f"Dropping categorical columns: {list(categorical_columns)}")
        data = data.drop(columns=categorical_columns)
    num_trees = int(reduction_config.get("num_of_trees", 100))
    print('num_trees ', num_trees)
    rf = RandomForestRegressor(n_estimators=num_trees, random_state=42)
    print('*'*50)
    #print(rf.estimators_) 
    print('target ', target)
    print('data ', data)
    rf.fit(data, target)
    feature_importances = pd.Series(rf.feature_importances_, index=data.columns)
    top_features = feature_importances.nlargest(num_features_to_keep).index
    print(f"Selected features based on tree importance: {list(top_features)}")
    reduced_data = data[top_features]

elif reduction_method == "PCA":
    print("\nApplying PCA for feature reduction.")
    n_components = num_features_to_keep
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    reduced_data = pd.DataFrame(reduced_data, columns=[f"PCA_{i+1}" for i in range(n_components)])
    print(f"Reduced data dimensions to {n_components} components.")

else:
    raise ValueError(f"Unknown feature reduction method: {reduction_method}")

# Display the reduced dataset
print("\nReduced DataFrame:")
print(reduced_data.head())


# Extract prediction type and algorithms
prediction_type = config["design_state_data"]["target"]["prediction_type"]
algorithms = config["design_state_data"]["algorithms"]

# Initialize models dynamically based on the prediction type and hyperparameters
models = []
gridSearchModels = []
param_grids = {}



for model_name, model_details in algorithms.items():
    # print('model_name',model_name)
    # print('model_details',model_details)

    if not model_details["is_selected"]:
        #print(model_details['is_selected'])
        continue  # Skip models that are not selected

    if prediction_type == "Regression" and "Regressor" in model_name:
        if model_name == "RandomForestRegressor":
            models.append(RandomForestRegressor(
                n_estimators=model_details.get("min_trees", 10),  # min_trees -> n_estimators
                max_depth=model_details.get("max_depth", None),
                min_samples_leaf=model_details.get("min_samples_per_leaf_min_value", 1),
                random_state=42
            ))
        elif model_name == "DecisionTreeRegressor":
            models.append(DecisionTreeRegressor(
                max_depth=model_details.get("max_depth", None),
                min_samples_leaf=model_details.get("min_samples_per_leaf_min_value", 1),
                random_state=42
            ))
        elif model_name == "LinearRegression":
            models.append(LinearRegression(
                 n_jobs=model_details.get("parallelism", None),
                 fit_intercept=True
            ))
        elif model_name == "RidgeRegression":
            models.append(Ridge(alpha=model_details.get("alpha", 1.0),
                    max_iter=model_details.get("max_iter", None),         
                    solver=model_details.get("solver", "auto"),          
                    random_state=42 
            ))
        elif model_name == "LassoRegression":
            models.append(Lasso(alpha=model_details.get("alpha", 0.1),
                    max_iter=model_details.get("max_iter", 1000),         
                    tol=model_details.get("tol", 1e-4),                  
                    random_state=42
            ))
        elif model_name == "ElasticNetRegression":
            models.append(ElasticNet(alpha=model_details.get("regularization_term", 0.1),  
                    l1_ratio=model_details.get("l1_ratio", 0.5),          
                    max_iter=model_details.get("max_iter", 1000),         
                    tol=model_details.get("tol", 1e-4),                  
                    random_state=42 
            ))
        elif model_name == "GBTRegressor":
            models.append(GradientBoostingRegressor(
                n_estimators=model_details.get("min_trees", 100),
                max_depth=model_details.get("max_depth", 3),
                random_state=42
            ))
        elif model_name == "xg_boost":
            models.append(xgb.XGBRegressor(
                n_estimators=model_details.get("min_trees", 100),
                max_depth=model_details.get("max_depth", 3),
                learning_rate=model_details.get("learning_rate", 0.1),
                random_state=42
            ))
        else:
            print(f"Skipping unsupported model: {model_name} for regression.")

    elif prediction_type == "Classification" and "Classifier" in model_name:
        if model_name == "RandomForestClassifier":
            models.append(RandomForestClassifier(
                n_estimators=model_details.get("min_trees", 10),
                max_depth=model_details.get("max_depth", None),
                min_samples_leaf=model_details.get("min_samples_per_leaf_min_value", 1),
                random_state=42
            ))
        elif model_name == "DecisionTreeClassifier":
            models.append(DecisionTreeClassifier(
                max_depth=model_details.get("max_depth", None),
                min_samples_leaf=model_details.get("min_samples_per_leaf_min_value", 1),
                random_state=42
            ))
        elif model_name == "LogisticRegression":
            models.append(LogisticRegression(C=model_details.get("C", 1.0),                       
                penalty=model_details.get("penalty", "l2"),          
                solver=model_details.get("solver", "lbfgs"),         
                l1_ratio=model_details.get("l1_ratio", None),        
                random_state=42                                      
            ))
        elif model_name == "SVM":
            models.append(SVC(C=model_details.get("C", 1.0),                   
                kernel=model_details.get("kernel", "rbf"),             
                gamma=model_details.get("gamma", "scale"),             
                tol=model_details.get("tol", 1e-3),                   
                max_iter=model_details.get("max_iter", -1),            
                random_state=42 
            ))
        elif model_name == "KNN":
            models.append(KNeighborsClassifier(n_neighbors=model_details.get("k_value", [5])[0],              
                weights="distance" if model_details.get("distance_weighting", False) else "uniform",  
                algorithm=model_details.get("neighbour_finding_algorithm", "auto"),  
                p=model_details.get("p_value", 2) 
            ))
        elif model_name == "GBTClassifier":
            models.append(GradientBoostingClassifier(
                n_estimators=model_details.get("min_trees", 100),
                max_depth=model_details.get("max_depth", 3),
                random_state=42
            ))
        elif model_name == "xg_boost":
            models.append(xgb.XGBClassifier(
                n_estimators=model_details.get("min_trees", 100),
                max_depth=model_details.get("max_depth", 3),
                learning_rate=model_details.get("learning_rate", 0.1),
                random_state=42
            ))
        elif model_name == "extra_random_trees":
            models.append(ExtraTreesClassifier(
                n_estimators=model_details.get("min_trees", 100),
                max_depth=model_details.get("max_depth", None),
                min_samples_leaf=model_details.get("min_samples_per_leaf_min_value", 1),
                random_state=42
            ))
        elif model_name == "LogisticRegression":
            models.append(LogisticRegression(max_iter=model_details.get("max_iter", 100),        
                solver=model_details.get("solver", 'lbfgs'),        
                C=model_details.get("min_regparam", 0.5),           
                penalty="elasticnet" if model_details.get("min_elasticnet", 0.5) > 0 else "l2", 
                l1_ratio=model_details.get("min_elasticnet", 0.5)
            ))
        elif model_name == "SGD":
            models.append(SGDClassifier(loss=model_details.get("loss", 'hinge'),         
                max_iter=model_details.get("max_iterations", 1000),  
                tol=model_details.get("tolerance", 1e-3),         
                alpha=model_details.get("alpha_value", 0.0001),   
                penalty=model_details.get("penalty", 'l2'),       
                l1_ratio=model_details.get("l1_ratio", 0.15),     
                learning_rate=model_details.get("learning_rate", 'invscaling'), 
                random_state=42
            ))
        elif model_name == "neural_network":
            models.append(MLPClassifier(hidden_layer_sizes=(model_details.get("hidden_layer_size", 100),),
                                        max_iter=model_details.get("max_iter", 200),
                                        random_state=42))
        else:
            print(f"Skipping unsupported model: {model_name} for classification.")

    else:
        print(f"Skipping incompatible model: {model_name} for prediction type: {prediction_type}")
print(models)
# Print initialized models with hyperparameters
# print("\nInitialized Models:")
for model in models:
    # Apply GridSearchCV
    grid_search = GridSearchCV(model, param_grids, cv=5, scoring='accuracy' if prediction_type == "Classification" else 'neg_mean_squared_error')
    #models.append((model_name, grid_search))
    gridSearchModels.append(grid_search)

# print(models)


target_column = config["design_state_data"]["target"]["target"]
#target = data.pop(target_column)
X = data.drop(columns=[target_column])  # Features
y = data[target_column]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit models and print best parameters and performance
for model_name, grid_search in zip(models, gridSearchModels):
    print('loop grid_search', grid_search)
    # print('model_name', model_name)
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best score for {model_name}: {grid_search.best_score_}")

    # If you want to test on a test dataset (X_test, y_test)
    y_pred = grid_search.predict(X_test)
    print('y_pred ', y_pred)
    
    if prediction_type == "Regression":
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error for {model_name}: {mse}")
    else:  # Classification
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_name}: {accuracy}")








