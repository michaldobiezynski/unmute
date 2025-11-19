import { useState, useCallback, useRef } from "react";

type MicrophoneAccessType = "unknown" | "granted" | "refused";

export const useMicrophoneAccess = () => {
  const [microphoneAccess, setMicrophoneAccess] =
    useState<MicrophoneAccessType>("unknown");

  const mediaStream = useRef<MediaStream | null>(null);

  const askMicrophoneAccess = useCallback(
    async (disableEchoCancellation: boolean = false) => {
      try {
        mediaStream.current = await window.navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: !disableEchoCancellation,
            autoGainControl: !disableEchoCancellation,
            noiseSuppression: !disableEchoCancellation,
          },
        });

        // Wait for audio tracks to be fully live before returning
        const audioTracks = mediaStream.current.getAudioTracks();
        if (audioTracks.length > 0) {
          const track = audioTracks[0];

          // If track is not yet live, wait for it
          if (track.readyState !== "live") {
            await new Promise<void>((resolve) => {
              const checkLive = () => {
                if (track.readyState === "live") {
                  resolve();
                } else {
                  setTimeout(checkLive, 10);
                }
              };
              checkLive();
            });
          }

          console.debug("MediaStream track is live and ready");
        }

        setMicrophoneAccess("granted");
        return mediaStream.current;
      } catch (e) {
        console.error(e);
        setMicrophoneAccess("refused");
        return null;
      }
    },
    []
  );

  return {
    microphoneAccess,
    askMicrophoneAccess,
    mediaStream,
  };
};
