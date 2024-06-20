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
## Dataset
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


## Achieve effect
<div align=center>
<img width="607" alt="微信图片_20240620183756" src="https://github.com/Jason040114/SMS-text-classification-model/assets/125139348/29434e76-dc62-405a-8d04-2a2a29c37ec5">
</div>

## Bibliography
[1] Zong Yun. Application Research on Public Security Anti-Fraud Technology. In Proceedings of the 36th China (Tianjin) 2022 IT, Network, Information Technology, Electronics, and Instrumentation Innovation Academic Conference. Tianjin Electronics Society, 2022:4. DOI:10.26914/c.cnkihy.2022.014973.<br>
[2] Wang Ming. Design and Implementation of an Early Warning System for Fraudulent SMS Based on Real-Time Streaming Technology. Software, 2015, 36(01):32-37.<br>
[3] Wen Xia. Building Intelligent Capabilities for Monitoring and Discovering Fraud-Related SMS to Support Precise Management of Unwanted Messages. Digital Communications World, 2023(01):166-168.<br>
[4] Song Donghan, Hu Maodi, Ding Jielan, et al. Research on Cross-Type Text Classification Technology Based on Multi-Task Learning. Data Analysis and Knowledge Discovery, 2024-06-20. http://kns.cnki.net/kcms/detail/10.1478.G2.20240305.1404.002.html.<br>
[5]You Chang, Huang Cheng, Tian Xuan, et al. Research on Fraudulent Website Detection and Classification Technology Based on Multidimensional Features. Journal of Sichuan University (Natural Science Edition), 2024-06-20. https://doi.org/10.19907/j.0490-6756.2024.040003.<br>
[6]Y. Kontsewaya, E. Antonov, and A. Artamonov, "Evaluating the Effectiveness of Machine Learning Methods for Spam Detection," in Procedia Computer Science, vol. XX, no. Y, pp. Z-W, 201X.<br>
[7] G. Luo, N. Shah, H. Khan, and H. Amin Ul, "Spam Detection Approach for Secure Mobile Message Communication Using Machine Learning Algorithms," Security and Communication Networks, vol. XX, no. Y, pp. Z-W, 201X.<br>
[8] V. B. L. Velammal and A. N. Aarthy, "Improvised Spam Detection in Twitter Data Using Lightweight Detectors and Classifiers," International Journal of Web-Based Learning and Teaching Technologies (IJWLTT), vol. XX, no. Y, pp. Z-W, 201X.<br>
[9] S. Palla, R. Dantu, and J. W. Cangussu, "Spam Classification Based on E-Mail Path Analysis," International Journal of Information Security and Privacy (IJISP), vol. XX, no. Y, pp. Z-W, 201X.<br>
[10] L. Xiang, G. Guo, Q. Li, C. Zhu, J. Chen, and H. Ma, "Spam Detection in Reviews Using LSTM-Based Multi-Entity Temporal Features," Intelligent Automation & Soft Computing, vol. XX, no. Y, pp. Z-W, 201X.<br>
[11] S. A. A. Ghaleb, M. Mohamad, S. Fadzli, and W. A. H. M. Ghanem, "E-mail Spam Classification Using Grasshopper Optimization Algorithm and Neural Networks," Computers, Materials & Continua, vol. XX, no. Y, pp. Z-W, 201X.<br>
[12] K. Pawar and M. Patil, "A Framework for Spam Filtering Security Evaluation," Software Engineering, vol. XX, no. Y, pp. Z-W, 201X.<br>

