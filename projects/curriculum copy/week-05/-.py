X = patsy.dmatrix('~ C(age) + C(education) + C(occupation) +C(race) + C(sex) + C(marital) + C(hr_per_week) + C(country)', df)
y = df['income'].values

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=77)

lr = LogisticRegression(solver='liblinear')
lr_model = lr.fit(X_train, y_train)

lr_ypred = lr_model.predict(X_test)

lr_cm = confusion_matrix(y_test, lr_ypred, labels=lr.classes_)
lr_cm = pd.DataFrame(lr_cm, columns=lr.classes_, index=lr.classes_)
lr_cm

print classification_report(y_test, lr_ypred, labels=lr.classes_)

cvs1 = cross_val_score(lr, X, y, cv=3, scoring='f1_weighted')
cvs1
cvs1.mean()