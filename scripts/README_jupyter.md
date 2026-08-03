# Running Jupyter Notebooks on a remote server setup
## SSH Port Forwarding Method
1. Start Jupyter on the Remote Server Log in to your remote server via SSH and start Jupyter without opening a browser:
```
jupyter notebook --no-browser --port=8888
```
You can also bind it to all interfaces if needed:
```
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```
2. Create an SSH Tunnel from Local Machine On your local machine, forward the remote port to a local port:
```
ssh -L 8888:localhost:8888 <REMOTE_USER>@<REMOTE_HOST>
```
3. Access in Browser Open your browser and go to:
```
http://localhost:8888
```
Enter the token shown in the remote terminal output.

## VS Code Remote Jupyter Connection
1. Start Jupyter Server on Remote Machine Run on the remote server:
```
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```
2. Get the Access URL Copy the URL with token from the terminal, e.g.:
```
http://<remote-ip>:8888/?token=<token>
```
3. Connect via VS Code
	1. Open VS Code (desktop or web).
	2. Press Ctrl+Shift+P (or Cmd+Shift+P on Mac) and select Jupyter: Specify Jupyter Server for Connections.
	3. Paste the copied URL.
	4. You can now run notebooks directly in VS Code connected to the remote kernel.

For multiple notebooks, use different ports or tools like sshuttle for VPN-like access.