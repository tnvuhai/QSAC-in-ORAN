***AdapShare: An RL-Based Dynamic Spectrum Sharing Solution for O-RAN***

**Brief Description**
AdapShare is an O-RAN-compatible Dynamic Spectrum Sharing (DSS) solution leveraging Reinforcement Learning (RL) for intent-based spectrum management, 
with the primary goal of minimizing resource surpluses or deficits in Radio Access Networks (RANs). By employing RL agents, AdapShare intelligently 
learns network demand patterns and uses them to allocate resources. We demonstrate the efficacy of AdapShare in the spectrum sharing scenario between 
4G Long Term Evolution (LTE) and 5G New Radio (NR) networks. We employ real-world LTE resource usage data collected using BladeRF Software Defined Radio 
(SDR) and Online Watcher of LTE (OWL), an open-source LTE sniffer, and synthetic NR usage data generated using Time-Series Generative Adversarial 
Network (TimeGAN). In this document, we describe the steps followed for data collection and the scripts used for data processing, demand predictions, 
and resource allocation.

**Requirements**
Software: Ubuntu 20.04, OWL, Anaconda3, MATLAB 2023b, Simulink.
Hardware: Ettus USRP B210 or BladeRF SDR board. 

**Data Collection and Processing**
*We compiled an extensive dataset of LTE scheduling information, collected at the National Institute of Standards and Technology (NIST) Gaithersburg 
campus between January and February 2023. 
*The data was collected from the downlink traffic at 2115 MHz (Band 4) using a free and open-source LTE control channel decoder called the Online Watcher 
of LTE (OWL), developed by IMDEA Networks Institute.
*For the collection of real-world LTE signals, we used a BladeRF SDR board to transmit the data to a PC running Ubuntu 20.04 and executing OWL to decode 
the signals. It's worth noting that instead of BladeRF, Ettus USRP B210 can also be used. You can find the details for the installation of Ettus USRP B210, 
BladeRF SDR and OWL in the document **Installation_Manual for OWL.pdf**.
*The raw data collected using OWL and BladeRF is available on request. The file includes the following data:
	· System Frame Number (SFN): internal timing of LTE (1 every frame = 10 ms)
	· Subframe Index from 0 to 9 (1 subframe = 1 ms)
	· Radio Network Temporary Identifier (RNTI) in decimal
	· Direction: 1 = downlink; 0 = uplink
	· Modulation and Coding Scheme (MCS) in 0 - 31
	· Number of allocated resource blocks in 0 - 110
	· Transport block size in bits
	· Transport block size in bits (code word 0), -1 if n/a
	· Transport block size in bits (code word 1), -1 if n/a
	· Downlink Control Message (DCI) message type. 
	· New data indicator toggle for codeword 0
	· New data indicator toggle for codeword 1
	· Hybrid Automatic Repeat Request (HARQ) process id
	· Narrowband Control Channel Element (NCCE) location of the DCI message
	· Aggregation level of the DCI message
	· Control Frame Indicator (CFI)
	· DCI correctness check.
*We extracted the System Frame Number (SFN), Subframe Index, and number of allocated resource blocks corresponding to DCI message type 8, which corresponds 
to the subframes carrying data. We filled in the missing data for the missing subframe indices, added the time of data collection, and compiled a time series 
of the PRB usage for LTE. We also downsampled the data from a granularity of 1 ms (per subframe) to 1 hour using the mean function. Run the 
python scripts listed below in the given order to obtain the data corresponding to the mentioned granularities:
	· **/Data/Transform BladeRF+OWL Data.py** (dataset available at https://datapub.nist.gov/od/id/mds2-3178)
	· **/Data/Extracting Data For Time Series Generation.py** (dataset available at https://datapub.nist.gov/od/id/mds2-3178)
	· **/Data/Generating Per Day Data.py** (dataset available at https://datapub.nist.gov/od/id/mds2-3178)
	· **/Data/Resampling Time Series Data.py** 
*Using the resampled LTE demand time series data, we generated synthetic NR demand data using TimeGAN. You can run the python script **Synthetic Data Generation-TimeGAN.py** 
to obtain synthetic NR data and LTE data using the collected LTE data. Note that multiple LTE and NR demand time series datasets are generated to train the 
RL agent.
*We used the MATLAB script **/Data/CDF_LTE_NR_Demand_Plot.m** to generate and compare the Cumulative Distribution Function (CDF) for collected LTE data and synthetically 
generated NR data. The goal is to highlight the similarity in demand distribution between the compiled LTE dataset and the synthetically generated NR dataset, 
thereby confirming the effectiveness of TimeGAN.

**Demand Prediction and Resource Allocation Using RL agent**
*We utilize the MATLAB script **RL_Resource_Allocation.mlx** to train the agent, either DDPG or TD3, and to save the trained agent for subsequent simulation. 
Additionally, the MATLAB scripts **ResourceAllocation_ddpg.m** and **ResourceAllocation_td3.m** can be employed to train and simulate DDPG and TD3 agents, 
respectively. These scripts also provide results corresponding to the quasi-static approach, which serves as the baseline for this work.

***If you have any queries, please feel free to reach out to Sneihil Gopal at sneihil.gopal@nist.gov.***

References:
[1] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press, 2018.

