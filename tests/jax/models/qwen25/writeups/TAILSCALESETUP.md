Here’s a simplified step-by-step guide to deploy a Koyeb service using Tailscale for secure SSH access:

### Step 1: Set Up Tailscale

1. **Create a Tailscale Account**:
   - Sign up or log in to [Tailscale](https://tailscale.com/).

2. **Generate Tailscale Auth Key**:
   - Go to the keys settings page in Tailscale.
   - Create a new key with **Reusable** and **Ephemeral** options selected.

### Step 2: Deploy Koyeb Service

1. **Create a New Service**:
   - Use the Koyeb CLI to create a new service with the Tailscale example:
   ```bash
   /root/.koyeb/bin/koyeb service create \
     --name tt-tailscale-ssh \
     --type web \
     --docker-image koyeb/tenstorrent-examples/tt-tailscale-ssh \
     --env TAILSCALE_AUTHKEY=YOUR_AUTH_KEY \
     --env NODE_NAME=tt-on-koyeb \
     your-app-name
   ```
   - Replace `YOUR_AUTH_KEY` with the key you generated.

### Step 3: Connect via Tailscale

1. **SSH into the Koyeb Instance**:
   - From any machine on your Tailscale network, connect using:
   ```bash
   ssh root@tt-on-koyeb
   ```

2. **Using Cursor or VS Code**:
   - Install the **Remote-SSH** extension in Cursor or VS Code.
   - Add a new SSH host: `root@tt-on-koyeb`.
   - Connect through the Tailscale network.

### Step 4: Verify and Manage

1. **Check Service Status**:
   - Use the Koyeb CLI to check the status of your service:
   ```bash
   /root/.koyeb/bin/koyeb service describe tt-vsc-tunnel
   ```

2. **Access Logs**:
   - To view logs, use:
   ```bash
   /root/.koyeb/bin/koyeb service logs tt-vsc-tunnel
   ```

### Benefits of This Approach

- **Secure**: No public SSH exposure; uses Tailscale's secure mesh network.
- **Simplified Management**: No need to manage SSH keys or configurations.
- **Official Support**: Utilizes Koyeb's official example for best practices.
- **Reliable**: Maintains stable connections even if the service restarts.

This approach streamlines the deployment process while enhancing security and ease of use. Let me know if you need any further assistance!
