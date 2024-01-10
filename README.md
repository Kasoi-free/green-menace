# green-menace
A repo for LinkedIn shitpost

Prerequisites:
1. A Cuda-compatible GPU with at least 16GB memory. Like RTX 3090 (Ti), 4060 Ti, 4070 Ti Super, 4080 or 4090.
2. Windows 11 with WSL. Check here on how to enable WSL: https://www.ridom.de/u/Windows_Subsystem_For_Linux.html

Steps:
1. In Windows, run Windows Update and restart
2. In Windows, open CMD or Powershell and run two commands:
wsl --update
wsl --install
3. Finish setting up your Ubuntu
4. In Ubuntu, run the following commands one after another:
sudo apt-get update && sudo apt-get upgrade && sudo apt-get install git-all  python3-pip
pip install diffusers invisible_watermark transformers accelerate safetensors --upgrade
5. Start the party with:
python3 green-menace.py

To end the party hit Ctrl+C