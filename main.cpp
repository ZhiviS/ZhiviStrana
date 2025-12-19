// main.cpp
/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <sstream> 
#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <csignal>
#include <algorithm>

#include <cuda_runtime.h> 

#if defined(_WIN32) || defined(_WIN64)
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#endif

// --------------------------------------
std::atomic<bool> Pause(false);
std::atomic<bool> Paused(false);
std::atomic<bool> stopMonitorKey(false);
int idxcount = 0;
double t_Paused = 0.0;
bool randomMode = false;
bool backupMode = false;

using namespace std;

VanitySearch* g_vanity_search_ptr = nullptr;
std::atomic<bool> g_shutdown_initiated(false);

void signalHandler(int signum) {
    if (!backupMode) {
        printf("\n"); 
        fflush(stdout); 
        exit(signum);
    }

    if (g_shutdown_initiated.exchange(true)) {
        exit(signum);
    }
    
    cout << "\n[!] Ctrl+C Detected. Shutting down gracefully, please wait...";
    cout.flush();
    
    if (g_vanity_search_ptr != nullptr) {
        g_vanity_search_ptr->endOfSearch = true;
    }
}

#if defined(_WIN32) || defined(_WIN64)
void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		if (_kbhit()) {
			char ch = _getch();
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}
}
#else
struct termios original_termios;
bool terminal_mode_changed = false;
void restoreTerminalMode() {
    if (terminal_mode_changed) {
        tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
    }
}
void setupRawTerminalMode() {
    tcgetattr(STDIN_FILENO, &original_termios);
    terminal_mode_changed = true;
    struct termios new_termios = original_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}
void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		char ch;
		if (read(STDIN_FILENO, &ch, 1) > 0) {
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}
}
#endif

// --------------------------------------

void printHelp() {
    printf("Usage: ./kk -r <bits> [-a <b58_addr> | -p <pubkey> | -i <file>] [options]\n\n");
    
    printf("Modes (choose one):\n");
    printf("  -a <b58_addr>       Find the private key for a P2PKH Bitcoin address.\n");
    printf("  -p <pubkey>         Find the private key for a specific public key (hex, compressed format only).\n");
    printf("  -i <file>           Search for a list of addresses or public keys from a file (one per line).\n\n");
    
    printf("Keyspace:\n");
    printf("  -r <bits>           Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).\n\n");
    
    printf("Options:\n");
    printf("  -R                  Activate random mode.\n");
    printf("  -b                  Enable backup mode to resume from last progress (not for random mode).\n");
    printf("  -G <ID>             Specify the GPU ID to use, default is 0.\n");
    printf("  -h, --help          Display this help message.\n\n");

    printf("Note: When using -i, all targets in the file must be of the same type (all addresses or all public keys).\n\n");
    
    printf("Technical Support: gitlab.com/8891689\n");
    exit(0);
}

int getInt(string name, char* v) {
	int r;
	try { r = std::stoi(string(v)); }
	catch (std::invalid_argument&) {
		fprintf(stderr, "[ERROR] Invalid %s argument, number expected\n", name.c_str());
		exit(-1);
	}
	return r;
}

bool loadBackup(int& idxcount, double& t_Paused, int gpuid) {
    string filename = "schedule_gpu" + to_string(gpuid) + ".dat";
    ifstream inFile(filename, std::ios::binary);
    if (inFile) {
        inFile.read(reinterpret_cast<char*>(&idxcount), sizeof(idxcount));
        inFile.read(reinterpret_cast<char*>(&t_Paused), sizeof(t_Paused));
        inFile.close();
        return true;
    }
    return false;
}

// ------------------- Main -------------------

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);

#if !(defined(_WIN32) || defined(_WIN64))
    atexit(restoreTerminalMode);
    setupRawTerminalMode();
#endif

    std::thread inputThread(monitorKeypress);
    Timer::Init();
    Secp256K1* secp = new Secp256K1();
    secp->Init();

    if (argc < 2) {
        printHelp();
    }
    
    string target_address;
    string target_pubkey;
    string input_filename;
    int bits = 0;
    int gpuId = 0; // GPU 0
    uint32_t maxFound = 65536 * 4;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp();
        } else if (arg == "-b") {
            backupMode = true;
        } else if (arg == "-R") {
            randomMode = true;
        } else if (arg == "-a") {
            if (i + 1 < argc) { target_address = argv[++i]; } 
            else { fprintf(stderr, "[ERROR] An address value is required after the -a parameter.\n"); exit(-1); }
        } else if (arg == "-p") {
            if (i + 1 < argc) { target_pubkey = argv[++i]; }
            else { fprintf(stderr, "[ERROR] A public key hex string is required after the -p parameter.\n"); exit(-1); }
        } else if (arg == "-i") {
            if (i + 1 < argc) { input_filename = argv[++i]; }
            else { fprintf(stderr, "[ERROR] A filename is required after the -i parameter.\n"); exit(-1); }
        } else if (arg == "-r") {
            if (i + 1 < argc) {
                bits = getInt((char*)"-r", argv[++i]);
                if (bits <= 0 || bits > 256) { fprintf(stderr, "[ERROR] -r value (number of bits) must be between 1 and 256.\n"); exit(-1); }
            } else { fprintf(stderr, "[ERROR] A numeric value is required after the -r parameter.\n"); exit(-1); }
        } else if (arg == "-G") {
            if (i + 1 < argc) gpuId = getInt((char*)"-G", argv[++i]);
        } else {
            fprintf(stderr, "[ERROR] Unknown parameter: %s\n", arg.c_str());
            printHelp();
        }
    }
    
    if ((target_address.empty() && target_pubkey.empty() && input_filename.empty()) || bits == 0) {
        fprintf(stderr, "[ERROR] A target source (-a, -p, or -i) and range (-r) must be specified.\n");
        printHelp();
    }
    if (!input_filename.empty() && (!target_address.empty() || !target_pubkey.empty())) {
        fprintf(stderr, "[ERROR] Cannot use -i with -a or -p. Please choose one target source.\n");
        printHelp();
    }
    if (!target_address.empty() && !target_pubkey.empty()) {
        fprintf(stderr, "[ERROR] Cannot use -a and -p at the same time. Please choose one.\n");
        printHelp();
    }
    if (backupMode && randomMode) {
        fprintf(stderr, "[ERROR] Backup mode (-b) cannot be used with random mode (-R).\n");
        exit(-1);
    }
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA error while checking for devices: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "[INFO] Please ensure NVIDIA drivers and CUDA toolkit are installed correctly.\n");
        exit(-1);
    }

    if (deviceCount == 0) {
        fprintf(stdout, "[INFO] No CUDA-enabled GPU was detected. Exiting.\n");
        exit(0);
    }

    if (gpuId >= deviceCount || gpuId < 0) {
        fprintf(stdout, "[INFO] Invalid GPU ID %d specified. No device detected with this ID.\n", gpuId);
        fprintf(stdout, "[INFO] Detected %d GPU(s). Valid IDs are from 0 to %d.\n", deviceCount, deviceCount - 1);
        exit(0);
    }


    vector<string> target_vector;
    string search_target_display; 

    if (!input_filename.empty()) {

        ifstream infile(input_filename.c_str());
        if (!infile.is_open()) {
            fprintf(stderr, "[ERROR] Could not open target file: %s\n", input_filename.c_str());
            exit(-1);
        }
        string line;
        while (getline(infile, line)) {

            if (!line.empty() && line.find_first_not_of(" \t\r\n") != string::npos) {
                target_vector.push_back(line);
            }
        }
        infile.close();
        
        if (target_vector.empty()) {
            fprintf(stderr, "[ERROR] Target file '%s' is empty or contains no valid lines.\n", input_filename.c_str());
            exit(-1);
        }
        
        printf("[+] Original targets from file: %zu\n", target_vector.size());
        std::sort(target_vector.begin(), target_vector.end());
        auto last = std::unique(target_vector.begin(), target_vector.end());
        target_vector.erase(last, target_vector.end());
        printf("[+] Unique targets after deduplication: %zu\n", target_vector.size());

        search_target_display = "from file '" + input_filename + "'";

    } else if (!target_pubkey.empty()) {

        target_vector.push_back(target_pubkey);
        search_target_display = target_pubkey;
    } else {

        target_vector.push_back(target_address);
        search_target_display = target_address;
    }

    BITCRACK_PARAM bitcrack, *bc;
    bc = &bitcrack;
    bc->ksStart.SetInt32(1);
    if (bits > 1) {
        bc->ksStart.ShiftL(bits - 1);
    }
    bc->ksFinish.SetInt32(1);
    bc->ksFinish.ShiftL(bits);
    bc->ksFinish.SubOne();
    bc->ksNext.Set(&bc->ksStart);

    if (backupMode) {
        if (loadBackup(idxcount, t_Paused, gpuId)) {
            printf("[+] Restoring from backup was successful. Starting batch: %d, Elapsed time: %.2f s.\n", idxcount, t_Paused);
        } else {
            printf("[+] Backup file not found. Will start from scratch.\n");
        }
    }
    
    printf("[+] KeyKiller v.007\n");
    if (!target_pubkey.empty()) {
        printf("[+] Search: %s [Public Key]\n", search_target_display.c_str());
    } else if (!input_filename.empty()) {
        printf("[+] Search: %zu targets %s\n", target_vector.size(), search_target_display.c_str());
    }
    else {
        printf("[+] Search: %s [P2PKH/Compressed]\n", search_target_display.c_str());
    }
    time_t now = time(NULL);
    printf("[+] Start %s", ctime(&now));
    if (randomMode) printf("[+] Random mode\n");
    printf("[+] Range (2^%d)\n", bits);
    printf("[+] from : 0x%s\n", bc->ksStart.GetBase16().c_str());
    printf("[+] to   : 0x%s\n", bc->ksFinish.GetBase16().c_str());
    fflush(stdout);

    VanitySearch* v = new VanitySearch(secp, target_vector, SEARCH_COMPRESSED, true, "", maxFound, bc);
    g_vanity_search_ptr = v; 
    vector<int> gpuIds = { gpuId };
    vector<int> gridSizes = { -1, 128 }; 
    
    v->Search(gpuIds, gridSizes);

    stopMonitorKey = true;
    if (inputThread.joinable()) {
        inputThread.join();
    }
    printf("\n");
    delete v;
    delete secp;
    return 0;
}
