import { useEffect, useRef } from "react";
import { ChatMessage } from "./chatHistory";
import { X } from "lucide-react";
import { createPortal } from "react-dom";

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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Format chat history as text
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

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isVisible && textareaRef.current) {
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
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
      <div
        className="bg-darkgray border-2 border-green shadow-lg w-full max-w-3xl max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center px-6 py-4 border-b border-lightgray">
          <h2 className="text-xl font-semibold text-white">
            Conversation Transcript
          </h2>
          <button
            onClick={onClose}
            className="text-lightgray hover:text-white transition-colors"
            aria-label="Close transcript"
          >
            <X size={24} />
          </button>
        </div>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={formatTranscript()}
          readOnly
          className="flex-1 w-full px-6 py-4 bg-background text-white resize-none focus:outline-none font-mono text-sm leading-relaxed"
          aria-label="Conversation transcript - readonly"
          style={{ minHeight: "400px" }}
        />

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

