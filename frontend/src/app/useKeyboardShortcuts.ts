import { useEffect, useState } from "react";

const ALLOW_DEV_MODE = false;

const useKeyboardShortcuts = () => {
  // local storage persistence disabled in case random users activate it accidentally
  // useLocalStorage("useDevMode", false)
  const [isDevMode, setIsDevMode] = useState(false);
  // useLocalStorage("showSubtitles", false)
  const [showSubtitles, setShowSubtitles] = useState(true);
  // useLocalStorage("showTranscript", false)
  const [showTranscript, setShowTranscript] = useState(false);

  const toggleSubtitles = () => {
    setShowSubtitles((prev) => !prev);
  };

  const toggleTranscript = () => {
    setShowTranscript((prev) => !prev);
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      // Don't toggle dev mode if the active element is an input field
      const isInputField =
        activeElement &&
        (activeElement.tagName === "INPUT" ||
          activeElement.tagName === "TEXTAREA" ||
          activeElement.getAttribute("contenteditable") === "true");

      if (
        ALLOW_DEV_MODE &&
        !isInputField &&
        (event.key === "D" || event.key === "d")
      ) {
        setIsDevMode((prev) => !prev);
      }
      if (!isInputField && (event.key === "S" || event.key === "s")) {
        toggleSubtitles();
      }
      if (!isInputField && (event.key === "T" || event.key === "t")) {
        toggleTranscript();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [toggleSubtitles, toggleTranscript]);

  return { isDevMode, showSubtitles, toggleSubtitles, showTranscript, toggleTranscript };
};

export default useKeyboardShortcuts;
