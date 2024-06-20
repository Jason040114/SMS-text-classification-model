# SMS-text-classification-model
This project is an SMS text classification model. This model can be used to **classify the incoming SMS text as scam SMS or non-scam SMS**.
## Operating environment
1. Python 3.7 or later<br>
2. PyTorch 1.9.0 or later<br>
3. Scikit-learn 0.24.2 or later<br>
4. NLTK 3.6.2 or later<br>
5. Pandas 1.3.0 or later<br>
## Run command
First, run main.py<br>
Then, code will be executed in sequence:
1. Rule-based scam SMS recognition model
2. Scams recognition model based on traditional machine learning
3. Scams recognition model based on Convolutional neural network (add PCA dimensionality reduction)
4. Scams recognition model based on convolutional neural network (without adding PCA dimension reduction)
CPU and memory consumption increase in turn
## dataset
The dataset contains 5,574 SMS texts and their corresponding classification labels, which are mainly used for SMS text classification tasks. Each record in the dataset contains two columns:<br>
1.sms: indicates the SMS content. The type is a string.<br>
2.label: indicates the category label. The type is an integer, where:<br>
&nbsp;&nbsp;```'0' means normal SMS (non-spam SMS)```<br>
&nbsp;&nbsp;```'1' indicates spam messages```<br>
<html xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">
<head>

<meta name=Generator content="Microsoft Excel">
<!--[if !mso]>

<!--.font0
	{color:#000000;
	font-size:12.0pt;
	font-family:宋体;
	font-weight:400;
	font-style:normal;
	text-decoration:none;}
.font1
	{color:#000000;
	font-size:11.0pt;
	font-family:宋体;
	font-weight:400;
	font-style:normal;
	text-decoration:none;}
br
	{mso-data-placement:same-cell;}
td
	{padding-top:1px;
	padding-left:1px;
	padding-right:1px;
	mso-ignore:padding;
	color:#000000;
	font-size:12.0pt;
	font-weight:400;
	font-style:normal;
	text-decoration:none;
	font-family:宋体;
	mso-generic-font-family:auto;
	mso-font-charset:134;
	mso-number-format:General;
	border:none;
	mso-background-source:auto;
	mso-pattern:auto;
	text-align:general;
	vertical-align:middle;
	white-space:nowrap;
	mso-rotate:0;
	mso-protection:locked visible;}
.et2
	{color:#000000;
	font-size:11.0pt;
	mso-generic-font-family:auto;
	mso-font-charset:134;}
-->

</head>
<body>
<!--StartFragment-->

sms | label
-- | --
Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat... | 0
Ok lar... Joking wif u oni... | 0
Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's | 1
U dun say so early hor... U c already then say... | 0
Nah I don't think he goes to usf, he lives around here though | 0


<!--EndFragment-->
</body>

</html>

## Model
1. **Based on rules**:According to daily experience, the keywords that may appear in fraud information are extracted and classified by direct judgment.
2. **Based on traditional machine learning**:TF-IDF is used to extract the feature vector from each short message. A naive Bayes classifier is used to classify short message texts.
3. **Convolutional neural network based**: The convolutional neural network is trained to process the feature vectors and find the appropriate parameters to complete the establishment of the model. (At the same time, PCA processing can reduce the dimension of the feature vector and greatly improve the speed, but the accuracy will be slightly decreased).

