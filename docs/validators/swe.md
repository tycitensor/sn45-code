# SWE Start


## Remote Server Setup

You should use a separate server from the one you run the validator on for this. This is to ensure security and avoid any potential issues.

### Setup Docker

Install docker: https://docs.docker.com/engine/install/ubuntu/
setup https://docs.docker.com/engine/daemon/remote-access/#configuring-remote-access-with-daemonjson with 0.0.0.0:2375

```bash
sudo systemctl edit docker.service
```

```bash
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// -H tcp://0.0.0.0:2375
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker.service
```

### Configure UFW

```bash
sudo ufw disable
```


### Crontab for docker registry
Create a file in `/etc/cron.daily/dockerio` with the following content:

```bash
#!/bin/bash
IPSET_NAME="dockerio"

# Clear existing IPs in the set
sudo ipset flush $IPSET_NAME

# Resolve required domains and add to ipset
for domain in registry-1.docker.io auth.docker.io cdn.docker.io; do
    for ip in $(dig +short $domain); do
        sudo ipset add $IPSET_NAME $ip
    done
done
sudo iptables -A OUTPUT -m set --match-set dockerio dst -p tcp --dport 443 -j ACCEPT
sudo iptables -A OUTPUT -m set --match-set dockerio dst -p tcp --dport 80 -j ACCEPT
sudo systemctl restart docker
```

Ensure the file is executable:
```bash
sudo chmod +x /etc/cron.daily/dockerio
```



### IPTables
The order of the rules is important.

```bash
sudo iptables -F
sudo iptables -t nat -F
sudo iptables -X
sudo iptables -A OUTPUT -p tcp --dport 2375 -j ACCEPT
sudo iptables -A INPUT -p tcp --sport 2375 -j ACCEPT

sudo iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
sudo iptables -A INPUT -p udp --sport 53 -j ACCEPT

sudo iptables -A INPUT -p tcp --sport 443 -j ACCEPT
iptables -I OUTPUT 1 -p tcp --dport 3000 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 3000 -j ACCEPT


sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT


# Allow outgoing SSH traffic (port 22)
sudo iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT

# Allow incoming SSH traffic (port 22)
sudo iptables -A INPUT -p tcp --sport 22 -j ACCEPT
sudo iptables -A OUTPUT -j DROP

sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

## Testing Docker Remote Access

From the server you are running the validator on (NOT THE ONE YOU RAN THESE COMMANDS ON), run the following command:

```bash
curl <docker-server-ip>:2375
```

that should return `{"message":"page not found"}`

Next to test further run:

```bash
DOCKER_HOST=tcp://<docker-server-ip>:2375 docker run --rm ubuntu bash -c "sleep 600"
```

While that command is running you should be able to go onto the docker server and see the container running.

```bash
docker ps
```

