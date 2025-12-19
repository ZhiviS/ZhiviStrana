<!-- Core Feature -->
<div align="center">
  <img src="https://img.shields.io/badge/Utility-GPU%20Satoshi%20Puzzle%20Solver-red?style=for-the-badge" alt="GPU Satoshi Puzzle Solver">
</div>

<!-- 🎯 Target & Acceleration -->
<div align="center">
  <img src="https://img.shields.io/badge/Target-Large%20Keyspaces%20(e.g.,%202%5E71)-orange?logo=bitcoin&logoColor=white" alt="Targets Large Bitcoin Keyspaces">
  <img src="https://img.shields.io/badge/Acceleration-NVIDIA%20CUDA-brightgreen?logo=nvidia" alt="NVIDIA CUDA Acceleration">
</div>

<!-- 🚀 Optimizations & Techniques -->
<div align="center">
  <img src="https://img.shields.io/badge/Optimization-Warp--Level%20Parallelism-purple" alt="Warp-Level Parallelism">
  <img src="https://img.shields.io/badge/Technique-Batch%20EC%20Operations-blue" alt="Batch EC Operations">
</div>

<!-- ⚙️ Features & Strategy -->
<div align="center">
  <img src="https://img.shields.io/badge/Search%20Modes-Address%20%7C%20Public%20Key-blue" alt="Address & Public Key Search Modes">
  <img src="https://img.shields.io/badge/Strategy-Sequential%20(Backup)%20%7C%20Random-lightgrey" alt="Sequential (with Backup) & Random Strategy">
</div>

<!-- Tech Stack -->
<div align="center">
  <img src="https://img.shields.io/badge/Language-C++%20%7C%20CUDA-blue?logo=cplusplus" alt="C++ and CUDA">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey?logo=linux&logoColor=black" alt="Platform Support">
</div>

<!-- Project Status & Activity -->
<div align="center">
  <a href="https://gitlab.com/8891689/KeyKiller-Cuda/-/commits/main" title="View latest commit"><img src="https://img.shields.io/gitlab/last_commit/8891689/KeyKiller-Cuda?gitlab_url=https://gitlab.com" alt="Last Commit"></a>
  <a href="https://gitlab.com/8891689/KeyKiller-Cuda" title="Star this project!"><img src="https://img.shields.io/gitlab/stars/8891689/KeyKiller-Cuda?gitlab_url=https://gitlab.com&style=social" alt="GitLab Stars"></a>
  <a href="https://gitlab.com/8891689/KeyKiller-Cuda/-/blob/main/LICENSE" title="Project License"><img src="https://img.shields.io/gitlab/license/8891689/KeyKiller-Cuda?gitlab_url=https://gitlab.com" alt="License"></a>
</div>




# KeyKiller Cuda


KeyKiller is the GPU-powered version of the KeyKiller project, designed to achieve extreme performance in solving Satoshi puzzles on modern NVIDIA GPUs. 
Leveraging CUDA, warp-level parallelism, and batch EC operations, KeyKiller CUDA pushes the limits of cryptographic key search.

1. The Secp256k1 algorithm is based on the excellent work of [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) ， [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul)，This implementation is based on modifications and improvements of the above implementation. Contributions are welcome! The algorithm has been significantly modified for CUDA. Special thanks to Jean-Luc Pons for his pioneering contributions to the cryptography community.

2. KeyKiller GPU-based solution to Satoshi's puzzle. This is an experimental project, Please look at it rationally! 

3. While KeyKiller CUDA is simple to use, it leverages massive GPU parallelism** to achieve extreme performance in elliptic curve calculations, compressed public keys, and Hash160 pipelines.

4. In theory, 4090 automatically configures the size, and the theoretical speed is about 6G, but this needs to be tested on the actual platform. Each platform environment is different, and the results obtained are also different. 
5. This program is still in the testing stage and may have unknown issues. It will continue to be improved and deeply optimized.

## Key Features

1. GPU Acceleration: Optimized for NVIDIA GPUs with full CUDA support.
2. Massive Parallelism: Tens of thousands of threads computing elliptic curve points and hash160 simultaneously.
3. Batch EC Operations: Efficient group addition and modular inversion with warp-level optimizations.
4. Grid/Batch Control: Use GPU execution with automatically configured parameters (number of threads per batch × number of points per batch).
5. Cross-Platform: Works on Linux and Windows .
6. -R command random mode does not slow down, high-speed calculation.
7. Incremental mode, with -b breakpoint save progress mode so you can continue working when you have time.

## User Manual
```bash
./kk -h
Usage: ./kk -r <bits> [-a <b58_addr> | -p <pubkey> | -i <file>] [options]

Modes (choose one):
  -a <b58_addr>       Find the private key for a P2PKH Bitcoin address.
  -p <pubkey>         Find the private key for a specific public key (hex, compressed format only).
  -i <file>           Search for a list of addresses or public keys from a file (one per line).

Keyspace:
  -r <bits>           Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).

Options:
  -R                  Activate random mode.
  -b                  Enable backup mode to resume from last progress (not for random mode).
  -G <ID>             Specify the GPU ID to use, default is 0.
  -h, --help          Display this help message.

Note: When using -i, all targets in the file must be of the same type (all addresses or all public keys).

Technical Support: github.com/8891689

```
## Options
- **-a**: Given a P2PKH Bitcoin address, crack its private key.
- **-p**: Given a public key, crack its private key. It must be a compressed public key.
- **-i**: Given Search for a list of addresses or public keys from a file (one per line).
- **-r**: range of search. Must be a power of two!Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).
- **-R**: Activate random mode.
- **-b**: Enable backup mode to resume from last progress (not for random mode).
- **-G**: Specify the GPU ID to use, default is 0.
-  **p**: Press the p key to pause your work and press it again to resume it.

## Example Output

Below is a sample run of KeyKiller for reference.

**RTX1030**

```bash
./kk -r 33 -a 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
[+] KeyKiller v.007
[+] Search: 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu [P2PKH/Compressed]
[+] Start Fri Sep 19 04:40:53 2025
[+] Range (2^33)
[+] from : 0x100000000
[+] to   : 0x1FFFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(256x256)
[+] Starting keys set in 0.02 seconds
[+] GPU 57.61 Mkey/s][Total 2^31.39][Prob 65.62%] [50% in seconds][Found 0]  

[!] (Add): 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
[!] (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9MDGKrXXQL647jj
[!] (HEX): 0x00000000000000000000000000000000000000000000000000000001A96CA8D8

```

**RTX1030**
```bash
./kk -r 33 -a 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu -R
[+] KeyKiller v.007
[+] Search: 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu [P2PKH/Compressed]
[+] Start Fri Sep 19 04:41:56 2025
[+] Random mode
[+] Range (2^33)
[+] from : 0x100000000
[+] to   : 0x1FFFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(384x256)
[+] Starting keys set in 0.03 seconds
[+] [GPU 56.93 Mkey/s][Total 2^31.94][Prob 9.6e+01%][50% in seconds][Found 0]  

[!] (Add): 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
[!] (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9MDGKrXXQL647jj
[!] (HEX): 0x00000000000000000000000000000000000000000000000000000001A96CA8D8

```
**RTX1030**
```bash
./kk -r 31 -p 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28
[+] KeyKiller v.007
[+] Search: 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28 [Public Key]
[+] Start Fri Sep 19 04:59:25 2025
[+] Range (2^31)
[+] from : 0x40000000
[+] to   : 0x7FFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(256x256)
[+] Starting keys set in 0.02 seconds
[+] GPU 100.04 Mkey/s][Total 2^29.17][Prob 56.25%] [50% in seconds][Found 0]  

[!] (Pub): 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28
[!] (WIF): Compressed:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M9SmFMSCA4jQRW
[!] (HEX): 0x000000000000000000000000000000000000000000000000000000007D4FE747

```

**RTX1030**

```bash
./kk -r 31 -p 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28 -R
[+] KeyKiller v.007
[+] Search: 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28 [Public Key]
[+] Start Fri Sep 19 04:57:39 2025
[+] Random mode
[+] Range (2^31)
[+] from : 0x40000000
[+] to   : 0x7FFFFFFF
[+] GPU: GPU #0 NVIDIA GeForce GT 1030 (3x128 cores) Grid(384x256)
[+] Starting keys set in 0.03 seconds
[+] [GPU 98.95 Mkey/s][Total 2^26.58][Prob 9.4e+00%][50% in seconds][Found 0]  

[!] (Pub): 0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28
[!] (WIF): Compressed:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M9SmFMSCA4jQRW
[!] (HEX): 0x000000000000000000000000000000000000000000000000000000007D4FE747

```

## Compile

```bash
make all

git clone https://gitlab.com/8891689/keykiller-cuda.git
```
## Local test based on 1030

1. Address and HASH160 Mode
```bash

| GPU               | Grid      | Speed (Mkeys/s) | Notes        |
| RTX1030           | 256,256   | 56 Mkeys/s    | My test      |
```

2. Public Key Mode
```bash

| GPU               | Grid      | Speed (Mkeys/s) | Notes        |
| RTX1030           | 256,256   | 99.6 Mkeys/s    | My test      |

```
# Advanced graphical user interface version


<div align="center">
  <img src="./images/KeyKiller_gui_1.png" alt="KeyKiller GUI Interface" width="850">
</div>



```
Usage: ./kk [-n <bits> | -r <min:max>] [-a <addr> | -h <hash> | -p <pub> | -i <file> ] [options]

Modes (choose one):
  -a <b58_addr>       Find private key for Address. Supports Prefixes (e.g., 1Bit...).
  -h <hash160>        Find private key for Hash160. Supports Hex Prefixes.
  -p <pubkey>         Find private key for Public Key. Supports Hex Prefixes.
  -i <file>           Search for a list of targets from a file (one per line).

Keyspace (choose one):
  -n <bits>           Set the bit range for the search (e.g., -n 71 for 2^71 to 2^71-1 , Bitcoin Puzzle #71 ).
  -r <hexA:hexB>      Search keys in the hexadecimal range from A to B (e.g., -r 400000000000000000:7fffffffffffffffff Bitcoin Puzzle #71 ).

Options:
  -R                  Activate random mode.
  -b                  Enable backup mode to resume from last progress (not for random mode).
                      * Note: When performing multiple backups, you need to rename the backup files in the specified directory before proceeding!
  -G <ID>             Specify the GPU ID to use, default is 0.
  -h, --help          Display this help message.

  Note:               When using -i, mixed types (Address, Hash160, PubKey) and Prefixes are supported.
                      Press 'p' at any time to pause/resume.

  Technical Support:  www.8891689.com
```


1. Some Linux systems may require granting permissions. 
Open the terminal, navigate to the program directory, and then type:

```
chmod +x keyelf_gui
```

Once completed, double-click to use the graphical interface version.

2. Some Windows systems may require the addition of variable values.

Once completed, double-click to use the graphical interface version.

3. The advanced graphical interface version is a paid service. You need to pay a membership fee to have the same password as other paid services in the open source library. This is because development requires financial support to maintain or develop more services. Thank you for your understanding!

4. If you have any questions, please contact me or visit the website directly: https://www.8891689.com/


# Sponsorship
If this project has been helpful or inspiring, please consider buying me a coffee. Your support is greatly appreciated. Thank you!

```
BTC: bc1qt3nh2e6gjsfkfacnkglt5uqghzvlrr6jahyj2k
ETH: 0xD6503e5994bF46052338a9286Bc43bC1c3811Fa1
DOGE: DTszb9cPALbG9ESNJMFJt4ECqWGRCgucky
TRX: TAHUmjyzg7B3Nndv264zWYUhQ9HUmX4Xu4
```

# ⚠️ Reminder:

This tool is for learning and research purposes only. Do not use it for illegal activities!

Decrypting someone else's private key is illegal and morally reprehensible. Please comply with local laws and regulations and use this tool only after understanding the associated risks.

The developer is not responsible for any direct or indirect financial losses or legal liabilities resulting from the use of this tool.