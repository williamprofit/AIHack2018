from sklearn import svm
from DataPreprocessor import DataPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from model import shuffleLists
from sklearn.model_selection import train_test_splitß

def test(e, o):
	total = len(e)
	correct = 0
	for i in range(total) :
		# print("predicted ")
		# print(e[i])
		# print("original ")
		# print(o[i])
		if e[i] == o[i]:
			correct += 1
	print("predicted ")
	print(correct)
	print("total")
	print(total)

def runLogisticRegression(data):
	X, y = data.getDataForCasualties()
	shuffleLists(X, y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
	scaler = StandardScaler()
	X_std = scaler.fit_transform(X_train)
	clf = LogisticRegression(random_state=0, solver='sag')
	lr = clf.fit(X_std, y_train)

	test(clf.predict(X_test), y_test)

def main():
	dataprep = DataPreprocessor()
	dataprep.preprocess()
	runLogisticRegression(dataprep)
	

if __name__ == '__main__':
	main()