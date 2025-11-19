declare module "opus-recorder" {
  interface RecorderOptions {
    // Either provide mediaTrackConstraints (and the library calls getUserMedia internally)
    // or provide sourceNode (if you already have a MediaStream)
    mediaTrackConstraints?: MediaStreamConstraints;
    sourceNode?: MediaStreamAudioSourceNode;
    encoderPath: string;
    bufferLength: number;
    encoderFrameSize: number;
    encoderSampleRate: number;
    maxFramesPerPage: number;
    numberOfChannels: number;
    recordingGain: number;
    resampleQuality: number;
    encoderComplexity: number;
    encoderApplication: number;
    streamPages: boolean;
  }

  export default class Recorder {
    constructor(options: RecorderOptions);
    start(): void;
    stop(): void;
    ondataavailable: (data: Uint8Array) => void;
    encodedSamplePosition: number;
  }
}


type DecoderWorker = Worker