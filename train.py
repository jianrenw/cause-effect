import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def feature_extractor():
    features = [
                # ('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                # ('A: Number of Unique Samples', 'A',
                #  f.SimpleTransform(transformer=f.count_unique)),
                # ('B: Number of Unique Samples', 'B',
                #  f.SimpleTransform(transformer=f.count_unique)),
                ('A: Normalized Entropy', 'A',
                 f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B',
                 f.SimpleTransform(transformer=f.normalized_entropy)),
                ('Pearson R', ['A',
                               'B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A', 'B'],
                 f.MultiColumnTransform(f.correlation_magnitude)),
                ('Entropy Difference', ['A', 'B'],
                 f.MultiColumnTransform(f.entropy_difference))
                ]
    combined = f.FeatureMapper(features)
    return combined


def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=50,
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=10,
                                                random_state=1))]
    # steps = [("extract_features", features),
    #          ("classify",
    #           GradientBoostingClassifier(loss='deviance',
    #                                      learning_rate=0.1,
    #                                      n_estimators=500,
    #                                      subsample=1.0,
    #                                      min_samples_split=8,
    #                                      min_samples_leaf=1,
    #                                      max_depth=9,
    #                                      init=None,
    #                                      random_state=1,
    #                                      max_features=None,
    #                                      verbose=0))]
    return Pipeline(steps)


def main():
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    target = data_io.read_train_target()

    print("Extracting features and training model")
    classifier = get_pipeline()
    classifier.fit(train, target.Target)

    print("Saving the classifier")
    data_io.save_model(classifier)


if __name__ == "__main__":
    main()
