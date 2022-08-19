# Order-book-imbalance
Can orderbook imbalances predict mid-price changes? Yes, there's a better than even chance.

Our research this time around looks into whether orderbook imbalances is predictive of mid-price changes. For this project we would be using a high frequency tick data for an undisclosed security. Our definition of orderbook imbalancce is (bid LO-ask LO)/(bid LO+ask LO). For every mid price change there would be a corresponding orderbook imbalance figure. We split the data into two parts, of which we fit a Logistic Regression onto the training data to predict the test data.

Our result show 60-70% prediction accuracy on the test data. Intuitively accuracy is higher when imbalance is bigger. We will expand our work here to other securities and asset class in the near future.

![image](https://user-images.githubusercontent.com/105033135/185676975-934f8560-3f9f-489d-b4d6-9e3cd459591b.png)

![image](https://user-images.githubusercontent.com/105033135/185677452-41cc5ae4-1144-465c-afd5-d6cab863c1d6.png)
