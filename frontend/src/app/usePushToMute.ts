import { useCallback, useEffect, useState } from "react";

interface UsePushToMuteOptions {
  /**
   * The keyboard key to use for push-to-mute (default: "Space")
   * Common options: "Space", "m", "Control", "Shift"
   */
  muteKey?: string;
  
  /**
   * Only enable mute functionality when connected
   */
  isEnabled?: boolean;
}

export const usePushToMute = (options: UsePushToMuteOptions = {}) => {
  const { muteKey = "Space", isEnabled = true } = options;
  const [isMuted, setIsMuted] = useState(false);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!isEnabled) return;
      
      // Prevent default behaviour for Space key to avoid page scrolling
      if (event.code === muteKey) {
        event.preventDefault();
        setIsMuted(true);
      }
    },
    [muteKey, isEnabled]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent) => {
      if (!isEnabled) return;
      
      if (event.code === muteKey) {
        event.preventDefault();
        setIsMuted(false);
      }
    },
    [muteKey, isEnabled]
  );

  useEffect(() => {
    if (!isEnabled) {
      setIsMuted(false);
      return;
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    // Clean up when unmounted or disabled
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [handleKeyDown, handleKeyUp, isEnabled]);

  // Manual toggle for button-based control
  const toggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  return {
    isMuted,
    toggleMute,
    setIsMuted,
  };
};

