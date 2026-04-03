# Multi-Device Setup Guide: UDP Audio Network Implementation

VoxAuth features a peer-to-peer (P2P) Secure Calling module designed to stream authenticated, isolated voice data over a local-area network (LAN). This guide explains how to properly set up the application so multiple devices can connect to each other.

## 1. Network Requirements

For two computers to communicate successfully using VoxAuth's Secure Call Engine:
- **Same Local Network:** Both computers must be connected to the same Wi-Fi router or wired network segment.
- **Firewall Rules:** The underlying `AudioServer` and `AudioClient` classes use **UDP port 5005** to exchange real-time audio streams. Ensure that your operating system's firewall allows incoming and outgoing UDP traffic on port 5005.

> [!WARNING]
> By default, Windows Defender Firewall may block python.exe from receiving inbound connections on new ports. You will likely see a firewall prompt the first time you activate "Host Mode". You *must* click **Allow Access** for the call functionality to work.

## 2. Launching VoxAuth for LAN Access

By default, Streamlit binds to `localhost` (127.0.0.1), which means the application interface can only be accessed from the machine it is running on. 

To allow other devices on your network to view the VoxAuth Dashboard:

1. Open a Command Prompt or Terminal in the VoxAuth directory.
2. Launch the app using the `--server.address` flag:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
3. Look at your terminal output. Streamlit will provide a **Network URL** (e.g., `http://192.168.1.50:8501`).
4. Other devices on your local network can open their web browsers and navigate to that Network URL to access the UI.

## 3. How to Make a Secure Call

### Step A: The Host starts the Server
1. The **first user** navigates to the *Secure Call* page.
2. Select **Host Call**.
3. Toggle the **Boot Secure Transmission Engine** switch ON.
4. The UI will display the Host's IP Address (e.g., `192.168.1.50`) and wait for an incoming connection. 

### Step B: The Joiner Connects
1. The **second user** navigates to the *Secure Call* page.
2. Select **Join Call**.
3. Under "Lookup target by Username or manually enter IP", you can:
   - Type the Host's `Target Username` and click **Lookup IP** to automatically retrieve their IP from the VoxAuth database.
   - Or, manually type the Host's IP into the **Target IP Address** field.
4. Toggle the **Boot Secure Transmission Engine** switch ON.

### Step C: Peer-to-Peer Authentication
Once both users have activated their engines, the UDP sockets link up:
- The `AudioClient` instantly begins sending `16kHz` live audio blocks every 1.5 seconds to the `AudioServer`.
- The `AudioServer` catches the caller's IP from the UDP datagram and dynamically sets it as the outgoing destination, initiating a 2-way stream.
- Audio is only transmitted after passing the local SepFormer AI isolation and ECAPA-TDNN biometric verification check! If the AI catches an unregistered voice, it will flag it as a threat and drop the UDP packet safely.
