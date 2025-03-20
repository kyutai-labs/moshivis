import { useState, useEffect, useCallback, useRef } from "react";
import { WSMessage } from "../../../protocol/types";
import { decodeMessage, encodeMessage } from "../../../protocol/encoder";

export const useSocket = ({
  onMessage,
  uri,
  onDisconnect: onDisconnectProp,
  imageUrl,
}: {
  onMessage?: (message: WSMessage) => void;
  uri: string;
  onDisconnect?: () => void;
  imageUrl?: string;
}) => {
  const lastMessageTime = useRef<null | number>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [imageSent, setImageSent] = useState(false);
  const [onConnectDone, setOnConnectDone] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const sendMessage = useCallback(
    (message: WSMessage) => {
      if (!socket) {
        console.log("socket not present");
        return false;
      }
      // audio message with no connection
      if (message.type == "audio" && !isConnected) {
        console.log("isConnected false on audio message, please wait for handshake");
        return false;
      }
      // otherwise send message
      socket.send(encodeMessage(message));
      return true;
    },
    [isConnected, socket],
  );
  useEffect(() => {
    async function sendImage() {
      console.log("image send", imageSent);
      console.log("image url", imageUrl);
      if (imageUrl && !imageSent) {
        const imageBytes = await fetchImageBytes(imageUrl);
        const sent = sendMessage({
          type: "image",
          data: imageBytes,
        });
        if (sent) {
          console.log("Image sent");
          setImageSent(true);
        }
      }
    }

    sendImage();
  }, [socket, onConnectDone, imageUrl, imageSent]);

  const onConnect = useCallback(() => {
    console.log("connected, now waiting for handshake.");
    setOnConnectDone(true);
  }, [setIsConnected, socket]);

  const onDisconnect = useCallback(() => {
    console.log("disconnected");
    if (onDisconnectProp) {
      onDisconnectProp();
    }
    setIsConnected(false);
  }, [onDisconnectProp]);

  const onMessageEvent = useCallback(
    (eventData: MessageEvent) => {
      lastMessageTime.current = Date.now();
      const dataArray = new Uint8Array(eventData.data);
      const message = decodeMessage(dataArray);
      if (message.type == "handshake") {
        console.log("Handshake received, let's rocknroll.");
        setIsConnected(true);
      }
      if (!onMessage) {
        return;
      }
      onMessage(message);
    },
    [onMessage, setIsConnected],
  );

  const start = useCallback(() => {
    const ws = new WebSocket(uri);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", onConnect);
    ws.addEventListener("close", onDisconnect);
    ws.addEventListener("message", onMessageEvent);
    setSocket(ws);
    console.log("Socket created", ws);
    lastMessageTime.current = Date.now();
  }, [uri, onMessage, onDisconnectProp]);

  const stop = useCallback(() => {
    setIsConnected(false);
    if (onDisconnectProp) {
      onDisconnectProp();
    }
    socket?.close();
    setSocket(null);
  }, [socket]);

  useEffect(() => {
    if (!isConnected) {
      return;
    }
    let intervalId = setInterval(() => {
      if (lastMessageTime.current && Date.now() - lastMessageTime.current > 10000) {
        console.log("closing socket due to inactivity", socket);
        socket?.close();
        onDisconnect();
        clearInterval(intervalId);
      }
    }, 500);

    return () => {
      lastMessageTime.current = null;
      clearInterval(intervalId);
    };
  }, [isConnected, socket]);

  return {
    isConnected,
    socket,
    sendMessage,
    start,
    stop,
  };
};

async function fetchImageBytes(imageUrl: string) {
  const response = await fetch(imageUrl);

  if (!response.ok) {
    throw new Error(`Failed to fetch image: ${response.statusText}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  return new Uint8Array(arrayBuffer);
}