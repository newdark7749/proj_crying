#include <iostream>
#include <fstream>
#include <Windows.h>
#include <mmsystem.h>
#include <stdlib.h>
#include <cstdlib>
#include <list>
#include <thread>
#include <string>
#include <time.h>
#include <vector>
#include <mutex>

#include <fdeep/fdeep.hpp>

#include <LibrosaCpp/test/wavreader.h>
#include <LibrosaCpp/librosa/librosa.h>

using namespace std;
using std::thread;
mutex mtx;

#pragma comment(lib, "winmm.lib")

#define TAG(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))


short int* waveIn = (short int*)malloc(sizeof(short int) * 16000 * 4);
short int* wave_4s = (short int*)malloc(sizeof(short int) * 16000 * 4);
short int* fr = (short int*)malloc(sizeof(short int) * 16000 * 1);
short int* temp_;

int ii = 0;
int clf_ii = 0;
char input = -1;

list<int>rlist;
list<int>alist;


const auto model = fdeep::load_model("cnn_m.json"); 
//const auto model = fdeep::load_model("simple_cnn.json");

int more_than_half(list<int>list);
void PlayRecord();
void SaveWavFile(const char*, WAVEFORMATEX*, PWAVEHDR);


struct wav_reader {
    FILE* wav;
    long data_pos;
    uint32_t data_length;
    uint32_t data_left;

    int format;
    int sample_rate;
    int bits_per_sample;
    int channels;
    int byte_rate;
    int block_align;

    int streamed;
};

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const& vec)
{
    int size = 0;
    for (auto& v : vec) {
        size += v.size();
    }

    std::vector<T> flattened;
    flattened.reserve(size);

    for (auto& v : vec) {
        for (auto& e : v) {
            flattened.push_back(e);
        }
    }

    return flattened;
}



void StartRecord()
{
    
 
    int sampleRate = 16000;
    const int NUMPTS = 16000 * 1;   // 1 seconds

    HWAVEIN    hWaveIn;
    MMRESULT   result;
    WAVEFORMATEX pFormat;
    pFormat.wFormatTag = WAVE_FORMAT_PCM;     // simple, uncompressed format
    pFormat.nChannels = 1;                    //  1=mono, 2=stereo
    pFormat.nSamplesPerSec = sampleRate;      // 8.0 kHz, 11.025 kHz, 22.05 kHz, and 44.1 kHz
    pFormat.nAvgBytesPerSec = sampleRate * 2;   // =  nSamplesPerSec × nBlockAlign
    pFormat.nBlockAlign = 2;                  // = (nChannels × wBitsPerSample) / 8
    pFormat.wBitsPerSample = 16;              //  16 for high quality, 8 for telephone-grade
    pFormat.cbSize = 0;

    // Specify recording parameters

    if(waveInOpen(&hWaveIn, WAVE_MAPPER, &pFormat,0L, 0L, WAVE_FORMAT_DIRECT)){
        cout << "Failed to open waveform input device." << endl;
    }
    WAVEHDR     WaveInHdr;

    // Set up and prepare header for input
    WaveInHdr.lpData = (LPSTR)fr;
    WaveInHdr.dwBufferLength = NUMPTS * 2; // 16000 * 1 (* 2)
    WaveInHdr.dwBytesRecorded = 0;
    WaveInHdr.dwUser = 0L;
    WaveInHdr.dwFlags = 0L;
    WaveInHdr.dwLoops = 0L;
    if (waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR))) {
        cout << "waveInPrepareHeader error" << endl;
        waveInClose(hWaveIn);
    }

    // Insert a wave input buffer
    if (waveInAddBuffer(hWaveIn, &WaveInHdr, sizeof(WAVEHDR))) {
        cout << "waveInAddBuffer error" << endl;
        waveInClose(hWaveIn);
    }



    // Commence sampling input
    result = waveInStart(hWaveIn);

    cout << "rec======" << ii << "s" << endl;
    

    // Wait until finished recording
    Sleep(1 * 1000);

    //mtx.lock();

    if (ii <= 3) {
        //cout << "func1, ii:" << ii << endl;
        memcpy(waveIn + (ii * 16000), fr, 16000 * 1 * (sizeof(short int)));


  
    }
    else {
        //cout << "func1,ii:"<<ii << endl;

        //cout << "waveinhdr.dwbytesrecored:" << WaveInHdr.dwBytesRecorded << endl;
        //cout << "length:" << WaveInHdr.dwBufferLength << endl;
        //cout << "waveIn arr size:" << (_msize(waveIn) / sizeof(*waveIn)) << endl;

        //mtx.lock();

        temp_ = waveIn;
        //waveIn = (short int*)realloc(waveIn, _msize(waveIn) + WaveInHdr.dwBytesRecorded);
        waveIn = (short int*)realloc(waveIn, _msize(waveIn) + _msize(fr));
        if (waveIn == 0)waveIn = temp_;
        

        
        memcpy(waveIn + (ii * 16000), fr, 16000 * 1 * (sizeof(short int)));
        
        //mtx.unlock();

        //cout << "fr m size:" << _msize(fr) << endl;
        //cout << "waveinhdr.dwbytesrecored:" << WaveInHdr.dwBytesRecorded << endl;
        //cout << "length:" << WaveInHdr.dwBufferLength << endl;
        //cout << "waveIn arr size:" << (_msize(waveIn) / sizeof(*waveIn)) << endl;

        //cout << "func1 memcpy done" << endl;




        WAVEHDR wavih;
        wavih.dwBufferLength = 16000 * 2 * ii;
        wavih.lpData = (LPSTR)waveIn;
        wavih.dwBytesRecorded = 0;
        wavih.dwUser = 0;
        wavih.dwFlags = 0;
        wavih.dwLoops = 0;
        wavih.reserved = 0;

        SaveWavFile("C:\\Users\\Administrator\\source\\repos\\tf\\tf\\real_total.wav", &pFormat, &wavih);
        
        //cout << "func1 save done" << endl;


    }

    

    waveInClose(hWaveIn);

    //mtx.unlock();
    // PlayRecord();


}
void func1() {
    while (true) {
        //cout << "<<func1>> ii:" << ii << endl;
        StartRecord();
        ii += 1;
        if (input != -1){
            free(waveIn);
            free(fr);
            break;
        }
    }
}

void func2() {
    //cout << "func2 start ii:" << ii << endl;
    while (true) {
        if (input != -1) {
            free(wave_4s);
            break;
        }

        if (ii >= 4) {
            cout << "<<<<<<clf time>>>>>>" << clf_ii <<"s" << endl;

            int sampleRate = 16000;

            WAVEFORMATEX pFormat;
            pFormat.wFormatTag = WAVE_FORMAT_PCM;     // simple, uncompressed format
            pFormat.nChannels = 1;                    //  1=mono, 2=stereo
            pFormat.nSamplesPerSec = sampleRate;      // 8.0 kHz, 11.025 kHz, 22.05 kHz, and 44.1 kHz
            pFormat.nAvgBytesPerSec = sampleRate * 2;   // =  nSamplesPerSec × nBlockAlign
            pFormat.nBlockAlign = 2;                  // = (nChannels × wBitsPerSample) / 8
            pFormat.wBitsPerSample = 16;              //  16 for high quality, 8 for telephone-grade
            pFormat.cbSize = 0;

            //mtx.lock();
            memcpy(wave_4s, waveIn+(16000*clf_ii), 16000 * 4 * (sizeof(short int)));
            
            //cout << "wave_4s msize:" << _msize(wave_4s)/(sizeof(short int)) << endl;

            WAVEHDR wih;
            wih.dwBufferLength = 16000 * 2 * 4;
            wih.lpData = (LPSTR)wave_4s;
            wih.dwBytesRecorded = 0;
            wih.dwUser = 0;
            wih.dwFlags = 0;
            wih.dwLoops = 0;
            wih.reserved = 0;

            //SaveWavFile("C:\\Users\\Administrator\\source\\repos\\tf\\tf\\real_total.wav", &pFormat, &wih);
            SaveWavFile("C:\\Users\\Administrator\\source\\repos\\tf\\tf\\thr_4s.wav", &pFormat, &wih);

            //mtx.unlock();

            //cout << "func2 save done" << endl;


            // mels extraction
            void* h_x = wav_read_open("C:\\Users\\Administrator\\source\\repos\\tf\\tf\\thr_4s.wav");
            int format, channels, sr, bits_per_sample;
            unsigned int data_length;
            int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
            if (!res)
            {
                cerr << "get ref header error: " << res << endl;
            }
            int samples = data_length * 8 / bits_per_sample;
            std::vector<int16_t> tmp(samples);
            res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
            if (res < 0)
            {
                cerr << "read wav file error: " << res << endl;
            } 
            std::vector<float> x(samples);
            std::transform(tmp.begin(), tmp.end(), x.begin(),
                [](int16_t a) {
                    return static_cast<float>(a) / 32767.f;
                });
            int n_fft = 400;
            int n_hop = 160;
            int n_mel = 40;
            int fmin = 0;
            int fmax = 8000;
            vector<vector<float>> mels = librosa::Feature::melspectrogram(x, sr, n_fft, n_hop, "hann", true, "reflect", 2.f, n_mel, fmin, fmax);
            //std::cout << "mel.dims: [" << mels.size() << "," << mels[0].size() << "]" << endl;

            //cout << "mel extract done" << endl;

            //flatten
            vector<float> mels_fl = flatten(mels);


            // predict
            auto result_1 = model.predict({
                fdeep::tensor(fdeep::tensor_shape(40,401,1),mels_fl)
                });

            float y_h = stof(fdeep::show_tensors(result_1).substr(6, 6));


            cout << "y_h:" << y_h << endl;



            // classify 
            float thr = 0.5;
            int det;
            if (y_h < thr) {
                det = 0;
                rlist.push_back(det);
            }
            else {
                det = 1;
                rlist.push_back(det);
            }


            int sDB = 0;

            for (int i = 62400; i < 64000; i++) {
                int ab = abs(wave_4s[i]);
                sDB += ab;
            }
            sDB = sDB / 1600;
            cout << ">>>>>>>>>sDB:" << sDB << endl;

            int dbTHRESHOLD = 2000;


            if (rlist.size() == 6) {
                rlist.pop_front();
                cout << "r_list:";
                for (int val : rlist)cout << val << " ";
                cout << endl;
                alist.push_back(more_than_half(rlist));
                if (alist.size() == 6) {
                    alist.pop_front();
                    cout << "a_list:";
                    for (int val2 : alist)cout << val2 << " ";
                    cout << endl;
                    if (more_than_half(alist) == 0) {
                        if (sDB >= dbTHRESHOLD) {
                            cout << "CRYING - take care a baby" << endl;
                        }
                    }
                    else {
                        if (sDB >= dbTHRESHOLD) {
                            cout << "WAIT" << endl;
                        }
                    }
                }
                else {
                    if (more_than_half(rlist) == 0) {
                        if (sDB >= dbTHRESHOLD) {
                            cout << "CRYING - take care a baby" << endl;
                        }
                    }
                    else {
                        if (sDB >= dbTHRESHOLD) {
                            cout << "WAIT" << endl;
                        }
                    }
                }
            }
            clf_ii++;
            
        }
        
        Sleep(1 * 1000);
    }
}


int main() {

    //StartRecord();
    thread t1(func1);
    //t1.detach();


    thread t2(func2);
    //t2.detach();
    
    cout << "If press any key, the record will stop and save to the file." << endl;
    
    cin >> input;
    t1.join();
    t2.join();


    Sleep(1 * 1000);
  

    return 0;
}

void PlayRecord()
{
    const int NUMPTS = 16000 * 4;   // 4 seconds
    int sampleRate = 16000;
    // 'short int' is a 16-bit type; I request 16-bit samples below
                                // for 8-bit capture, you'd    use 'unsigned char' or 'BYTE' 8-bit types

    HWAVEIN  hWaveIn;

    WAVEFORMATEX pFormat;
    pFormat.wFormatTag = WAVE_FORMAT_PCM;     // simple, uncompressed format
    pFormat.nChannels = 1;                    //  1=mono, 2=stereo
    pFormat.nSamplesPerSec = sampleRate;
    pFormat.nAvgBytesPerSec = sampleRate * 2;   // = nSamplesPerSec * n.Channels * wBitsPerSample/8
    pFormat.nBlockAlign = 2;                  // = n.Channels * wBitsPerSample/8
    pFormat.wBitsPerSample = 16;              //  16 for high quality
    pFormat.cbSize = 0;

    // Specify recording parameters

    waveInOpen(&hWaveIn, WAVE_MAPPER, &pFormat, 0L, 0L, WAVE_FORMAT_DIRECT);

    WAVEHDR      WaveInHdr;
    // Set up and prepare header for input
    WaveInHdr.lpData = (LPSTR)waveIn;
    WaveInHdr.dwBufferLength = NUMPTS * 2;
    WaveInHdr.dwBytesRecorded = 0;
    WaveInHdr.dwUser = 0L;
    WaveInHdr.dwFlags = 0L;
    WaveInHdr.dwLoops = 0L;
    waveInPrepareHeader(hWaveIn, &WaveInHdr, sizeof(WAVEHDR));

    HWAVEOUT hWaveOut;
    cout << "playing..." << endl;
    waveOutOpen(&hWaveOut, WAVE_MAPPER, &pFormat, 0, 0, WAVE_FORMAT_DIRECT);
    waveOutWrite(hWaveOut, &WaveInHdr, sizeof(WaveInHdr)); // Playing the data
    Sleep(4 * 1000); //Sleep for as long as there was recorded

    waveInClose(hWaveIn);
    waveOutClose(hWaveOut);
}


int more_than_half(list<int>list) {

    int fst = 0;
    int snd = 0;
    for (int val : list) {
        if (val == 0)fst += 1;
        if (val == 1)snd += 1;
    }

    if (fst > snd) return 0;
    else return 1;

}

void SaveWavFile(const char* filename, WAVEFORMATEX* format, PWAVEHDR WaveHeader)
{
    // output stream을 할당
    ofstream ostream;
    // 파일 열기
    ostream.open(filename, fstream::binary);

    int subchunk1size = 16;
    int byteRate = format->nSamplesPerSec * format->nBlockAlign;
    int subchunk2size = WaveHeader->dwBufferLength * format->nChannels;
    int chunksize = (36 + subchunk2size);


    // wav파일 구조체대로 작성한다.
    ostream.seekp(0, ios::beg);
    // chunk id
    ostream.write("RIFF", 4);
    // chunk size (36 + SubChunk2Size))
    ostream.write((char*)&chunksize, 4);
    // format
    ostream.write("WAVE", 4);
    // subchunk1ID
    ostream.write("fmt ", 4);
    // subchunk1size (무압축 PCM이면 16 고정)
    ostream.write((char*)&subchunk1size, 4);
    // AudioFormat (무압축 PCM이면 1 고정)
    ostream.write((char*)&format->wFormatTag, 2);
    // NumChannels
    ostream.write((char*)&format->nChannels, 2);
    // sample rate  
    ostream.write((char*)&format->nSamplesPerSec, 4);
    // byte rate (SampleRate * block align)
    ostream.write((char*)&byteRate, 4);
    // block align
    ostream.write((char*)&format->nBlockAlign, 2);
    // bits per sample
    ostream.write((char*)&format->wBitsPerSample, 2);
    // subchunk2ID
    ostream.write("data", 4);
    // subchunk2size (NumSamples * nBlockAlign)  
    ostream.write((char*)&subchunk2size, 4);
    // 실제 음악 데이터 작성  
    ostream.write(WaveHeader->lpData, WaveHeader->dwBufferLength);
    // 파일 닫기
    ostream.close();
}

static uint32_t read_tag(struct wav_reader* wr) {
    uint32_t tag = 0;
    tag = (tag << 8) | fgetc(wr->wav);
    tag = (tag << 8) | fgetc(wr->wav);
    tag = (tag << 8) | fgetc(wr->wav);
    tag = (tag << 8) | fgetc(wr->wav);
    return tag;
}

static uint32_t read_int32(struct wav_reader* wr) {
    uint32_t value = 0;
    value |= fgetc(wr->wav) << 0;
    value |= fgetc(wr->wav) << 8;
    value |= fgetc(wr->wav) << 16;
    value |= fgetc(wr->wav) << 24;
    return value;
}

static uint16_t read_int16(struct wav_reader* wr) {
    uint16_t value = 0;
    value |= fgetc(wr->wav) << 0;
    value |= fgetc(wr->wav) << 8;
    return value;
}

static void skip(FILE* f, int n) {
    int i;
    for (i = 0; i < n; i++)
        fgetc(f);
}

void* wav_read_open(const char* filename) {
    struct wav_reader* wr = (struct wav_reader*)malloc(sizeof(*wr));
    memset(wr, 0, sizeof(*wr));

    if (!strcmp(filename, "-"))
        wr->wav = stdin;
    else
        wr->wav = fopen(filename, "rb");
    if (wr->wav == NULL) {
        free(wr);
        return NULL;
    }

    while (1) {
        uint32_t tag, tag2, length;
        tag = read_tag(wr);
        if (feof(wr->wav))
            break;
        length = read_int32(wr);
        if (!length || length >= 0x7fff0000) {
            wr->streamed = 1;
            length = ~0;
        }
        if (tag != TAG('R', 'I', 'F', 'F') || length < 4) {
            fseek(wr->wav, length, SEEK_CUR);
            continue;
        }
        tag2 = read_tag(wr);
        length -= 4;
        if (tag2 != TAG('W', 'A', 'V', 'E')) {
            fseek(wr->wav, length, SEEK_CUR);
            continue;
        }
        // RIFF chunk found, iterate through it
        while (length >= 8) {
            uint32_t subtag, sublength;
            subtag = read_tag(wr);
            if (feof(wr->wav))
                break;
            sublength = read_int32(wr);
            length -= 8;
            if (length < sublength)
                break;
            if (subtag == TAG('f', 'm', 't', ' ')) {
                if (sublength < 16) {
                    // Insufficient data for 'fmt '
                    break;
                }
                wr->format = read_int16(wr);
                wr->channels = read_int16(wr);
                wr->sample_rate = read_int32(wr);
                wr->byte_rate = read_int32(wr);
                wr->block_align = read_int16(wr);
                wr->bits_per_sample = read_int16(wr);
                if (wr->format == 0xfffe) {
                    if (sublength < 28) {
                        // Insufficient data for waveformatex
                        break;
                    }
                    skip(wr->wav, 8);
                    wr->format = read_int32(wr);
                    skip(wr->wav, sublength - 28);
                }
                else {
                    skip(wr->wav, sublength - 16);
                }
            }
            else if (subtag == TAG('d', 'a', 't', 'a')) {
                wr->data_pos = ftell(wr->wav);
                wr->data_length = sublength;
                wr->data_left = wr->data_length;
                if (!wr->data_length || wr->streamed) {
                    wr->streamed = 1;
                    return wr;
                }
                fseek(wr->wav, sublength, SEEK_CUR);
            }
            else {
                skip(wr->wav, sublength);
            }
            length -= sublength;
        }
        if (length > 0) {
            fseek(wr->wav, length, SEEK_CUR);
        }
    }
    fseek(wr->wav, wr->data_pos, SEEK_SET);
    return wr;
}

void wav_read_close(void* obj) {
    struct wav_reader* wr = (struct wav_reader*)obj;
    if (wr->wav != stdin)
        fclose(wr->wav);
    free(wr);
}

int wav_get_header(void* obj, int* format, int* channels, int* sample_rate, int* bits_per_sample, unsigned int* data_length) {
    struct wav_reader* wr = (struct wav_reader*)obj;
    if (format)
        *format = wr->format;
    if (channels)
        *channels = wr->channels;
    if (sample_rate)
        *sample_rate = wr->sample_rate;
    if (bits_per_sample)
        *bits_per_sample = wr->bits_per_sample;
    if (data_length)
        *data_length = wr->data_length;
    return wr->format && wr->sample_rate;
}

int wav_read_data(void* obj, unsigned char* data, unsigned int length) {
    struct wav_reader* wr = (struct wav_reader*)obj;
    int n;
    if (wr->wav == NULL)
        return -1;
    if (length > wr->data_left && !wr->streamed) {
        int loop = 1;
        if (loop) {
            fseek(wr->wav, wr->data_pos, SEEK_SET);
            wr->data_left = wr->data_length;
        }
        length = wr->data_left;
    }
    n = fread(data, 1, length, wr->wav);
    wr->data_left -= length;
    return n;
}