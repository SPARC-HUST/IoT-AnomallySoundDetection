import string
import numpy as np 
import socket
import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-lh","--host",
                    help="IP addresses receiving data")
parser.add_argument("-p", "--port",
                    type=int,
                    help="Port")
parser.add_argument("-t", "--time",
                    type=int,
                    help="Audio file duration")
parser.add_argument("-f", "--folder",
                    help="Folder containing the audio file")
args = parser.parse_args()

class UdpReceiver:
    HOST = '0.0.0.0'
    DATAPACKSIZE = 128
    VOICEIPNUM = 1
    localPort = 0
    recSeconds = 0
    decodeTbl = [0xEA80,0xEB80,0xE880,0xE980,0xEE80,0xEF80,0xEC80,0xED80,0xE280,0xE380,0xE080,0xE180,0xE680,0xE780,0xE480,0xE580,
            0xF540,0xF5C0,0xF440,0xF4C0,0xF740,0xF7C0,0xF640,0xF6C0,0xF140,0xF1C0,0xF040,0xF0C0,0xF340,0xF3C0,0xF240,0xF2C0,
            0xAA00,0xAE00,0xA200,0xA600,0xBA00,0xBE00,0xB200,0xB600,0x8A00,0x8E00,0x8200,0x8600,0x9A00,0x9E00,0x9200,0x9600,
            0xD500,0xD700,0xD100,0xD300,0xDD00,0xDF00,0xD900,0xDB00,0xC500,0xC700,0xC100,0xC300,0xCD00,0xCF00,0xC900,0xCB00,
            0xFEA8,0xFEB8,0xFE88,0xFE98,0xFEE8,0xFEF8,0xFEC8,0xFED8,0xFE28,0xFE38,0xFE08,0xFE18,0xFE68,0xFE78,0xFE48,0xFE58,
            0xFFA8,0xFFB8,0xFF88,0xFF98,0xFFE8,0xFFF8,0xFFC8,0xFFD8,0xFF28,0xFF38,0xFF08,0xFF18,0xFF68,0xFF78,0xFF48,0xFF58,
            0xFAA0,0xFAE0,0xFA20,0xFA60,0xFBA0,0xFBE0,0xFB20,0xFB60,0xF8A0,0xF8E0,0xF820,0xF860,0xF9A0,0xF9E0,0xF920,0xF960,
            0xFD50,0xFD70,0xFD10,0xFD30,0xFDD0,0xFDF0,0xFD90,0xFDB0,0xFC50,0xFC70,0xFC10,0xFC30,0xFCD0,0xFCF0,0xFC90,0xFCB0,
            0x1580,0x1480,0x1780,0x1680,0x1180,0x1080,0x1380,0x1280,0x1D80,0x1C80,0x1F80,0x1E80,0x1980,0x1880,0x1B80,0x1A80,
            0x0AC0,0x0A40,0x0BC0,0x0B40,0x08C0,0x0840,0x09C0,0x0940,0x0EC0,0x0E40,0x0FC0,0x0F40,0x0CC0,0x0C40,0x0DC0,0x0D40,
            0x5600,0x5200,0x5E00,0x5A00,0x4600,0x4200,0x4E00,0x4A00,0x7600,0x7200,0x7E00,0x7A00,0x6600,0x6200,0x6E00,0x6A00,
            0x2B00,0x2900,0x2F00,0x2D00,0x2300,0x2100,0x2700,0x2500,0x3B00,0x3900,0x3F00,0x3D00,0x3300,0x3100,0x3700,0x3500,
            0x0158,0x0148,0x0178,0x0168,0x0118,0x0108,0x0138,0x0128,0x01D8,0x01C8,0x01F8,0x01E8,0x0198,0x0188,0x01B8,0x01A8,
            0x0058,0x0048,0x0078,0x0068,0x0018,0x0008,0x0038,0x0028,0x00D8,0x00C8,0x00F8,0x00E8,0x0098,0x0088,0x00B8,0x00A8,
            0x0560,0x0520,0x05E0,0x05A0,0x0460,0x0420,0x04E0,0x04A0,0x0760,0x0720,0x07E0,0x07A0,0x0660,0x0620,0x06E0,0x06A0,
            0x02B0,0x0290,0x02F0,0x02D0,0x0230,0x0210,0x0270,0x0250,0x03B0,0x0390,0x03F0,0x03D0,0x0330,0x0310,0x0370,0x0350]
   


    def VoiceIPRecord(self):
        buff  = np.short(np.array(range(self.DATAPACKSIZE)))
        self.HOST = args.host  #socket.gethostname() 
        self.DATAPACKSIZE = 128
        self.localPort = args.port
        self.recSeconds = args.time
        self.prefix = args.folder
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.bind((self.HOST,self.localPort))
        # GetConfig()
        

        # self.voIPSettings['IPAddress'] = '10.0.0.1'
        # self.voIPSettings['Port']= 16384
        # self.voIPSettings['Prefix']= 'D:/AudioProcessing/New folder/test1/voip' # wav file recording path
        
        self.voIPWrite =  WaveWrite(self.prefix,self.recSeconds)
            
        while True:
            data, remoteEP = sock.recvfrom(140)
            print(data)
            if data[:4] != b'BCOM':
                continue
            if data[7] != 0:
                continue        
            for j in range(0, self.DATAPACKSIZE):
                buff[j] = self.decodeTbl[data[12+j]]
            self.voIPWrite.Write(buff) 	        	


class WaveWrite:
    f = ''     
    filePrefix = ''  
    waveSeconds = 0
    limitCount = 0
    writeCount = 0

    def __init__(self, prefix, seconds):
        self.waveSeconds = seconds
        self.filePrefix = prefix
        self.limitCount = seconds * (8000// UdpReceiver.DATAPACKSIZE)
        self.Open()

    def WavHeader(self,sampleRate, bitsPerSample, channels): # samplingFreq,  sampleBit, channels
        datasize = sampleRate * self.waveSeconds* channels * bitsPerSample // 8# = 5274984
        header_file = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
        header_file += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
        header_file += bytes("WAVE",'ascii')                                              # (4byte) File type
        header_file += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
        header_file += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
        header_file += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
        header_file += (channels).to_bytes(2,'little')                                    # (2byte)
        header_file += (sampleRate).to_bytes(4,'little')                                  # (4byte)
        header_file += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
        header_file += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
        header_file += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
        header_file += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
        header_file += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
        return header_file   
# Create new file and write header
    def Open(self):
        x = datetime.datetime.now()
        fileName = self.filePrefix + x.strftime("%Y%m%d%H%M%S") + ".wav"
        self.writeCount = 0
        self.f = open(fileName,"wb")
        self.f.write(self.WavHeader(8000, 16, 1))   
# Write data           
    def Write(self,data):
        for i in range(0, UdpReceiver.DATAPACKSIZE):
            self.f.write(data[i])
        self.writeCount += 1
        if self.writeCount >= self.limitCount:
            self.f.close()   
            self.Open()
# 
RecordAudio = UdpReceiver()
RecordAudio.VoiceIPRecord()



      
  
