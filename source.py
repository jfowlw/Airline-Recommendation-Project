import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocessing(filename):
    dataset = pd.read_csv(filename)
    
    # Fill in missing values with -1
    dataset.overall_rating[dataset.overall_rating.isnull()]=-1
    dataset.seat_comfort_rating[dataset.seat_comfort_rating.isnull()]=-1
    dataset.cabin_staff_rating[dataset.cabin_staff_rating.isnull()]=-1
    dataset.food_beverages_rating[dataset.food_beverages_rating.isnull()]=-1
    dataset.inflight_entertainment_rating[dataset.inflight_entertainment_rating.isnull()]=-1
    dataset.ground_service_rating[dataset.ground_service_rating.isnull()]=-1
    dataset.wifi_connectivity_rating[dataset.wifi_connectivity_rating.isnull()]=-1
    dataset.value_money_rating[dataset.value_money_rating.isnull()]=-1
    dataset.fillna(-1)
    
    # Transform Categorical Variables into Vectors
    cols_to_transform = ["airline_name", "cabin_flown", "author_country", "type_traveller", "overall_rating", 
                         "seat_comfort_rating", "cabin_staff_rating", "food_beverages_rating", "inflight_entertainment_rating",
                         "ground_service_rating", "wifi_connectivity_rating", "value_money_rating"]
    ds_new = pd.get_dummies(dataset, columns = cols_to_transform )
    dataset = ds_new
    
    # TFIDF Vectorizer to replace content with TFIDF scores
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 250)
    content_vector = vectorizer.fit_transform(dataset['content'])
    content_features = vectorizer.get_feature_names()
    content_vector = content_vector.toarray()
    content_array = pd.DataFrame(data = np.squeeze(np.asarray(content_vector)),
                                 columns = content_features)
    
    ds_new = pd.concat([dataset, content_array], axis = 1)
    dataset = ds_new
    return dataset

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                    train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):
    # Random Forest Classifier for binary classification of data
    clf = RandomForestClassifier(max_features=75, n_estimators=50, n_jobs=-1)
    clf.fit(features, target)
    return clf

def main():
    # Load the csv file into pandas dataframe
    dataset = preprocessing('train.csv')
    final_test = preprocessing('test.csv')

    test_headers = list(final_test.columns.values)
    final_headers = list(dataset.columns.values)
    
    #Remove features that will not be used in the classifier
    final_headers.remove("recommended")
    final_headers.remove("route")
    final_headers.remove("link")
    final_headers.remove("author")
    final_headers.remove("aircraft")
    final_headers.remove("title")
    final_headers.remove("date")
    final_headers.remove("content")
    for header in final_headers:
        if header in test_headers == False:
            final_headers.remove(header)  
        elif header not in test_headers:
            final_headers.remove(header)      
    final_headers.remove('airline_name_jeju-air')
    final_headers.remove('airline_name_loganair')
    final_headers.remove('airline_name_skybus')
    final_headers.remove('author_country_Morocco')
    final_headers.remove('author_country_Uzbekistan')
    final_headers.remove('author_country_Venezuela')
    final_headers.remove('airline_name_bhutan-airlines')
    final_headers.remove('airline_name_solomon-airlines')
    final_headers.remove('author_country_Angola')
    final_headers.remove('author_country_Aruba')
    final_headers.remove('author_country_Bermuda')
    final_headers.remove('author_country_Brunei')
    final_headers.remove('author_country_Dominica')
    final_headers.remove('author_country_Georgia')
    final_headers.remove('author_country_Guam')
    final_headers.remove('aircraft')
    final_headers.remove('bag')
    final_headers.remove('route')
    
    # train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, final_headers[1:len(final_headers)], "recommended")
    # trained_model = random_forest_classifier(train_x, train_y)
    
    trained_model_final = random_forest_classifier(dataset[final_headers[1:len(final_headers)]], dataset["recommended"])
    # predictions = trained_model_final.predict(dataset[final_headers[1:len(final_headers)]])
    
    # predictions = trained_model.predict(test_x)
    # print ("ROC Score :: ", roc_auc_score(test_y, predictions))
    # print ("ROC Score :: ", roc_auc_score(dataset["recommended"], predictions))
    
    final_predictions = trained_model_final.predict(final_test[final_headers[1:len(final_headers)]])
    
    filename = "nocontent_predictions.csv"
    headers = "id,recommended"
    f = open(filename, "w")
    f.write(headers)
    for index, row in final_test.iterrows():
        f.write("\n" + str(row["id"]) + "," + str(final_predictions[index]))

    f.close()
    
main()