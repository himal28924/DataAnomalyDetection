import numpy as np
import matplotlib.pyplot as plt

# Since I have already done project like this for wind data in my machine learning project ,
# there I already had data from wind company but now I will have to create a new function
# that will give me time series data which has some anomaly
# As asked in mail , season pattern , I will create this using math sin function as it illustriate seasonal pattern behaivour
#

def simulate_timeSeries_data(length=1000, noise_level=0.2, anomaly_rate=0.05):

    t = np.arange(length)

    data = np.sin(2 * np.pi * t / 100)  # Seasonal pattern

    # Adding random noises # In default of 0.2
    noise = noise_level * np.random.randn(length)
    data += noise

    # Random spikes , adding 5 and -5 , cause normally sin goes from -1 to 1 ,now the anolomies can be seen clearly
    anomalies = np.random.rand(length) < anomaly_rate
    data[anomalies] += np.random.choice([-5, 5], size=np.sum(anomalies))  # Random large deviations

    return data, anomalies


# Z-score-based Anomaly Detection
# To flag anomalies I am using z score , meaning z scorer of a point  , that are far above thereshold,
# I would consider the anomalies

def detect_anomalies_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)

    z_scores = (data - mean) / std
    anomalies = np.abs(z_scores) > threshold

    return anomalies


# Simulate the data
data_stream, true_anomalies = simulate_timeSeries_data()

# Detect anomalies using Z-score
detected_anomalies = detect_anomalies_zscore(data_stream)

# Plotting the data stream and anomalies
plt.figure(figsize=(10, 6))
plt.plot(data_stream, label='Data Stream')
plt.scatter(np.where(true_anomalies), data_stream[true_anomalies], color='red', label='True Anomalies', zorder=5)
plt.scatter(np.where(detected_anomalies), data_stream[detected_anomalies], color='green', label='Detected Anomalies',
            marker='x', zorder=5)
plt.title('Simulated Timer series Data ')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# In conclusion when  running the code , most of the anomalies were detected but some were not , which can be justified in real life too.
# TO make it better I can :
# - Use alogrithm to handle concept drift
# _ In plot use real time data , like streaming
#  - While creating data set I can use time sleep to make it better