## SWE Start

Set the `LLM_AUTH_KEY` environment variable to a random string. This MUST be random, do not copy it.

```
export LLM_AUTH_KEY=1234567890
```

Install docker: https://docs.docker.com/engine/install/ubuntu/
setup https://docs.docker.com/engine/daemon/remote-access/#configuring-remote-access-with-daemonjson with 0.0.0.0:2375

setup ufw

```bash
sudo ufw allow 2375/tcp
sudo ufw allow 22/tcp

sudo ufw enable
```

## setup squid

```bash
# Install Squid
sudo apt install squid -y

# Edit the config file
sudo nano /etc/squid/squid.conf
```

```bash
# Allow specific domains
acl allowed_sites dstdomain .docker.io .docker.com
http_access allow allowed_sites
# Deny everything else
http_access deny all
```

## setup crontab

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



## setup iptables 
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


