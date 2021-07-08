# Awesome Codecs for Polar Codes

# Description
This repo implements various decoders for polar codes under AWGN channel. The implemented decoders include
* Successive Cancellation (SC) Decoder
* Successive Cancellation List (SCL) Decoder
* Fast SC Decoder 
* Fast SCL Decoder
* Quantized SC Decoder with Precomputed LookUp Table (LUT)
* Quantized SCL Decoder with Precomputed LookUp Table (LUT)
* Quantized FastSC Decoder with Precomputed LookUp Table (LUT)
* Quantized FastSCL Decoder with Precomputed LookUp Table (LUT)
* Uniformly Quantized SC Decoder
* Uniformly Quantized SCL Decoder 

The decoders are all implemented by C++11 with -o3 complier configuration for speed. The Cpp codes are easy to read without 
fancy optimization skills like SIMD or multi-threading. However, the speed is quite acceptable for research propose. Please 
have fun with it !

# Requirement
* numpy
* matplot
* pickle
* torchtracer
* scipy
* tqdm


# Installation
To use the simulation platform, you need to first compile the Cpp codes for the quantizer generation algorithms (MMI, MinDistortion), 
the decoders and the encoders, which locate in /Quantizer, /PolarDecoder, /PolarEncoder respectively.

We recommend you to enter these directory and read the README.md in these directory to compile and install the package. The Cpp codes are 
organized in different classes and wrapped with Python interface therefore we can direct call them in our Python codes.

# Usage
All BER/BLER simulation entry starts with main*
* mainFPDecoder.py: simulation entry for float point decoders (SC, SCL, FastSC, FastSCL, CASAL)
* mainQuantizedDecoder_ContinuousDomain.py: simulation entry for uniformly quantized decoders (UniformLSOptSC, UniformLSOptSCL)
* mainQuantizedDecoder_LLRDomain.py: simulation entry for Minimum Distortion Quantized Decoder
* mainQuantizedDecoder_ProbabilityDomain.py: simulation entry for quantized decoders with different approaches (MMI, MsIB, DegradeMerge) to quantize the bit-channel of polar codes

All LookUp Table generation entry starts with GenerateLookUpTable:
* GenerateLookUpTable_LLRDomain.py: generate Minimum Distortion Quantizers in quantized decoding in LLR domain
* GenerateLookUpTable_ProbabilityDomain.py: generate quantizers (MMI/MsIB/DegradeMerge) in  quantized decoding in probability domain

Because the LookUp Table generation process is relatively time-consuming, we separate the LookUp Table generation with the quantized decoding with MMI, MsIB, 
DegradeMerge, MinDistortion quantizers. 

For uniformly quantized decoders, the quantizer generation is quite fast therefore we merge the quantizer generation codes into the simulation codes

Before running simulation for quantized decoding with MMI, MsIB, DegradeMerge, MinDistortion quantizers, you should first run LookUp Table generation codes first and then run the simulation codes. 
If you run simulation for uniformly quantized decoders, you should just run the simulation code only and the quantizers will be first generated for simulation. 

# Examples
* Run simulation of Minimum Distortion Quantized Decoding
```bash
python3 GenerateLookUpTable_LLRDomain.py --N 128 --QDecoder 16 --QChannelUniform 128 -QChannelCompressed 16 --DesignSNRdB 3.0 
python3 mainQuantizedDecoder_LLRDomain.py --N 128 --A 32 --L 8 --DecoderType SCL-LUT --isCRC no --QChannelUniform 128 --QDecoder 16 --QChannel 16 --DesignSNRdB 3.0
```
* Run simulation of MMI/MsIB/DegradeMerge Quantized Decoding
```bash
python3 GenerateLookUpTable_ProbabilityDomain.py --N 128 --QDecoder 16 --QChannelUniform 128 -QChannelCompressed 16 --DesignSNRdB 3.0 --Quantizer MMI # --Quantizer MsIB --Quantizer DegradeMerge
python3 mainQuantizedDecoder_ProbabilityDomain.py --N 128 --A 32 --L 8 --DecoderType SCL-LUT --isCRC no --QChannelUniform 128 --QDecoder 16 --QChannel 16 --DesignSNRdB 3.0
```

* Run simulation of Uniformly Quantized Decoding
```bash
python3 mainQuantizedDecoder_ContinuousDomain.py --N 128 --A 32 --L 8 --DecoderType SCL-LUT --isCRC no --QChannelUniform 128 --QDecoder 16 --QChannel 16 --DesignSNRdB 3.0
```

Commend line parameters are enabled for the above codes, you can get access to the meaning of each parameter by running for example
```bash
python3 mainQuantizedDecoder_ContinuousDomain.py --help
```

# Related Papers
* Z. Cao, H. Zhu, Y. Zhao and D. Li, "Nonuniform Quantized Decoder for Polar Codes With Minimum Distortion Quantizer," in IEEE Communications Letters, vol. 25, no. 3, pp. 835-839, March 2021, doi: 10.1109/LCOMM.2020.3041902.

# TODO
* Add support of Windows platform compilation of cpp codes. Currently, the compilation of cpp codes are Ok with Linux or MacOS