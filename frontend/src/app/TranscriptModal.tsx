import { useEffect, useRef, useState } from "react";
import { ChatMessage } from "./chatHistory";
import { X, Copy, Check } from "lucide-react";
import { createPortal } from "react-dom";
import clsx from "clsx";

interface TranscriptModalProps {
  chatHistory: ChatMessage[];
  isVisible: boolean;
  onClose: () => void;
}

const TranscriptModal = ({
  chatHistory,
  isVisible,
  onClose,
}: TranscriptModalProps) => {
  const contentRef = useRef<HTMLDivElement>(null);
  const [isCopied, setIsCopied] = useState(false);
  const [showNotification, setShowNotification] = useState(false);

  // Format chat history as text for copying
  const formatTranscript = () => {
    if (chatHistory.length === 0) {
      return "No conversation yet. Start talking to see the transcript here.";
    }

    return chatHistory
      .map((message) => {
        const label = message.role === "user" ? "You" : "Assistant";
        return `${label}: ${message.content}`;
      })
      .join("\n\n");
  };

  // Copy transcript to clipboard
  const handleCopyTranscript = async () => {
    const transcript = formatTranscript();
    try {
      await navigator.clipboard.writeText(transcript);
      setIsCopied(true);
      setShowNotification(true);
      setTimeout(() => setIsCopied(false), 2000);
      setTimeout(() => setShowNotification(false), 3000);
    } catch (error) {
      console.error("Failed to copy transcript:", error);
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isVisible && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [chatHistory, isVisible]);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isVisible) {
        onClose();
      }
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isVisible, onClose]);

  if (!isVisible) {
    return null;
  }

  const modalContent = (
    <div
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      {/* Copy notification */}
      {showNotification && (
        <div className="fixed top-8 left-1/2 transform -translate-x-1/2 bg-green text-black px-6 py-3 rounded-lg shadow-lg z-[60] flex items-center gap-2 animate-pulse">
          <Check size={20} />
          <span className="font-medium">Transcript copied to clipboard!</span>
        </div>
      )}
      
      <div
        className="bg-darkgray border-2 border-green shadow-lg w-full max-w-3xl max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center px-6 py-4 border-b border-lightgray">
          <h2 className="text-xl font-semibold text-white">
            Conversation Transcript
          </h2>
          <div className="flex items-center gap-3">
            <button
              onClick={handleCopyTranscript}
              className="flex items-center gap-2 text-lightgray hover:text-green transition-colors"
              aria-label="Copy transcript to clipboard"
            >
              {isCopied ? (
                <>
                  <Check size={20} />
                  <span className="text-sm">Copied!</span>
                </>
              ) : (
                <>
                  <Copy size={20} />
                  <span className="text-sm">Copy</span>
                </>
              )}
            </button>
            <button
              onClick={onClose}
              className="text-lightgray hover:text-white transition-colors"
              aria-label="Close transcript"
            >
              <X size={24} />
            </button>
          </div>
        </div>

        {/* Transcript content */}
        <div
          ref={contentRef}
          className="flex-1 w-full px-4 py-4 bg-background overflow-y-auto"
          style={{ minHeight: "400px" }}
        >
          {chatHistory.length === 0 ? (
            <p className="text-lightgray text-center italic">
              No conversation yet. Start talking to see the transcript here.
            </p>
          ) : (
            <div className="flex flex-col gap-3">
              {chatHistory.map((message, index) => (
                <div key={`${message.role}-${index}`} className="flex flex-col gap-1">
                  <span
                    className={clsx(
                      "font-semibold text-sm",
                      message.role === "user" ? "text-blue-400" : "text-green"
                    )}
                  >
                    {message.role === "user" ? "You" : "Assistant"}:
                  </span>
                  <p
                    className={clsx(
                      "whitespace-pre-wrap break-words text-base leading-relaxed",
                      message.role === "user" ? "text-blue-100" : "text-white"
                    )}
                  >
                    {message.content}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-lightgray text-lightgray text-xs text-right">
          Press ESC to close
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
};

export default TranscriptModal;

