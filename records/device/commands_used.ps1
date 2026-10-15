# PowerShell snippets used (assume $PORT="COM7")
$PORT="COM7"

# Copy JSONs to device
mpremote connect $PORT fs mkdir /models
mpremote connect $PORT fs mkdir /models/A44_ET
mpremote connect $PORT fs mkdir /models/A45_RF
mpremote connect $PORT fs mkdir /models/A16_MLP
mpremote connect $PORT fs mkdir /models/A8_GMM
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\artifacts\exports\A44_ET\preprocess.json" :/models/A44_ET/preprocess.json
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\artifacts\exports\A45_RF\preprocess.json" :/models/A45_RF/preprocess.json
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\artifacts\exports\A16_MLP\preprocess.json" :/models/A16_MLP/preprocess.json
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\artifacts\exports\A8_GMM\preprocess.json" :/models/A8_GMM/preprocess.json
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\artifacts\exports\A8_GMM\gmm.json"        :/models/A8_GMM/gmm.json

# Push runtime scripts
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\model_rt_gmm.py" :/model_rt_gmm.py
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\feat_a8.py"     :/feat_a8.py
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\a8_live_gmm.py" :/a8_live_gmm.py
mpremote connect $PORT fs cp "C:\Users\ramin\Desktop\bearing-ml\a8_loop_gmm.py" :/a8_loop_gmm.py

# Run live tests
mpremote connect $PORT exec "import a8_live_gmm as t; t.run_once()"
mpremote connect $PORT exec "import a8_loop_gmm as t; t.run_loop()"