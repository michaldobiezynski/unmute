import { useState, useCallback, useRef } from "react";

type MicrophoneAccessType = "unknown" | "granted" | "refused";

export const useMicrophoneAccess = () => {
  const [microphoneAccess, setMicrophoneAccess] =
    useState<MicrophoneAccessType>("unknown");

  const mediaStream = useRef<MediaStream | null>(null);

  const askMicrophoneAccess = useCallback(async (disableEchoCancellation: boolean = false) => {
    try {
      mediaStream.current = await window.navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: !disableEchoCancellation,
          autoGainControl: !disableEchoCancellation,
          noiseSuppression: !disableEchoCancellation,
        }
      });
      setMicrophoneAccess("granted");
      return mediaStream.current;
    } catch (e) {
      console.error(e);
      setMicrophoneAccess("refused");
      return null;
    }
  }, []);

  return {
    microphoneAccess,
    askMicrophoneAccess,
    mediaStream,
  };
};
