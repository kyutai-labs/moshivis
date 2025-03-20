import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useMemo, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { ImageGallery } from "../../components/ImageGallery/ImageGallery";
import { useModelParams, importantSettingsHaveChanged } from "../Conversation/hooks/useModelParams";
import { ModelParams } from "../Conversation/components/ModelParams/ModelParams";
import { env } from "../../env";
import { useUserEmail } from "./hooks/useUserEmail";
import { Input } from "../../components/Input/Input";
import { getAPIClient } from "./api/client";

type Status = "connecting" | "in_queue" | "has_credentials" | "error" | "no_queue" | "idle" | "bypass";


function getFloatFromStorage(val: string | null) {
  return (val == null) ? undefined : parseFloat(val)
}

function getIntFromStorage(val: string | null) {
  return (val == null) ? undefined : parseInt(val)
}

function getBoolFromStage(val: string | null) {
  return (val == 'true') ? true : ((val == 'false') ? false : undefined)
}

export const Queue: FC = () => {
  const [searchParams] = useSearchParams();
  let queueId = searchParams.get("queue_id");
  if (!queueId) {
    queueId = 'talktomoshi';
  }
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [sessionAuthId, setSessionAuthId] = useState<string | null>(null);
  const [workerAddr, setWorkerAddr] = useState<string | null>(null);
  const [workerAuthId, setWorkerAuthId] = useState<string | null>(null);
  const [currentPosition, setCurrentPosition] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [shouldConnect, setShouldConnect] = useState<boolean>(false);
  let default_image_url = sessionStorage.getItem("imageUrl");
  const modelParams = useModelParams({
    textTemperature: getFloatFromStorage(sessionStorage.getItem("textTemperature")),
    textTopk: getIntFromStorage(sessionStorage.getItem("textTopk")),
    audioTemperature: getFloatFromStorage(sessionStorage.getItem("audioTemperature")),
    audioTopk: getIntFromStorage(sessionStorage.getItem("audioTopk")),
    padMult: getFloatFromStorage(sessionStorage.getItem("padMult")),
    repetitionPenalty: getFloatFromStorage(sessionStorage.getItem("repetitionPenalty")),
    repetitionPenaltyContext: getIntFromStorage(sessionStorage.getItem("repetitionPenaltyContext")),
    imageResolution: getIntFromStorage(sessionStorage.getItem("imageResolution")),
    gateInfluence: getFloatFromStorage(sessionStorage.getItem("gateInfluence")),
    displayColor: getBoolFromStage(sessionStorage.getItem("displayColor")),
    centerCrop: getBoolFromStage(sessionStorage.getItem("centerCrop")),
    imageUrl: (default_image_url == null) ? undefined : default_image_url
  });
  const modalRef = useRef<HTMLDialogElement>(null);

  let def_user_email = sessionStorage.getItem("userEmail");
  const { userEmail, setUserEmail, error: emailError, validate } = useUserEmail(!!overrideWorkerAddr, (def_user_email == null) ? '' : def_user_email);

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  // enable eruda in development
  useEffect(() => {
    if (env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if (env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage, setShouldConnect]);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if (worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const onConnect = useCallback(async () => {
    if (!validate(userEmail)) {
      return;
    }
    await startProcessor();
    const hasAccess = await getMicrophoneAccess();
    if (hasAccess) {
      setShouldConnect(true);
    }
  }, [setShouldConnect, startProcessor, userEmail, getMicrophoneAccess, validate]);

  const status: Status = useMemo(() => {
    if (overrideWorkerAddr) {
      return "bypass";
    }
    if (!queueId) {
      return "no_queue";
    }
    if (error) {
      return "error";
    }
    if (!shouldConnect) {
      return "idle";
    }
    if (workerAddr && workerAuthId) {
      return "has_credentials";
    }
    if (!sessionId || !sessionAuthId) {
      return "connecting";
    }
    return "in_queue";
  }, [queueId, sessionId, sessionAuthId, workerAddr, workerAuthId, currentPosition, hasMicrophoneAccess, error, shouldConnect]);

  const client = useMemo(() => {
    return getAPIClient(env.VITE_QUEUE_API_PATH)
  }, [env.VITE_QUEUE_API_PATH]);

  useEffect(() => {
    if (!shouldConnect) {
      return;
    }
    if (status !== "connecting" || !queueId) {
      return;
    }
    client.addUser(queueId)
      .then(({ session_id, session_auth_id }) => {
        setSessionId(session_id);
        setSessionAuthId(session_auth_id);
        console.log("Added user to queue", session_id, session_auth_id);
      })
      .catch((e) => {
        setError(e.message);
        console.error(e);
      });
  }, [queueId, client, status, shouldConnect]);

  useEffect(() => {
    if (!sessionId || !sessionAuthId) {
      return;
    }
    if (status === "has_credentials") {
      return;
    }
    let isQuerying = false;
    let intervalId: number | null = null;
    const checkUser = () => {
      if (isQuerying) {
        return;
      }
      isQuerying = true;
      client.checkUser(sessionId, sessionAuthId)
        .then(({ worker_addr, worker_auth_id, current_position }) => {
          setCurrentPosition(current_position);
          if (worker_addr && worker_auth_id) {
            setWorkerAddr(worker_addr);
            setWorkerAuthId(worker_auth_id);
            if (intervalId !== null) {
              clearInterval(intervalId);
            }
          }
        })
        .catch((e) => {
          if (intervalId !== null) {
            clearInterval(intervalId);
          }
          setError(e.message);
          console.error(e);
        }).finally(() => {
          isQuerying = false;
        });
    }
    intervalId = setInterval(checkUser, 400);
    return () => {
      if (intervalId !== null) {
        clearInterval(intervalId);
      }
    };
  }, [sessionId, sessionAuthId, client, setCurrentPosition, setWorkerAddr, setWorkerAuthId, status, setError]);


  if (status === "bypass" && hasMicrophoneAccess && audioContext.current && worklet.current) {
    return (
      <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        {...modelParams}
      />
    );
  }

  if (status === "has_credentials" && workerAddr && audioContext.current && workerAuthId && sessionId && sessionAuthId && worklet?.current) {
    return (
      <Conversation
        email={userEmail}
        workerAddr={overrideWorkerAddr ?? workerAddr}
        workerAuthId={workerAuthId}
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        sessionId={sessionId}
        sessionAuthId={sessionAuthId}
        onConversationEnd={() => {
          setWorkerAddr(null);
          setWorkerAuthId(null);
          setSessionId(null);
          setSessionAuthId(null);
          setShouldConnect(false);
        }}
        {...modelParams}
      />
    );
  }
  return (
    <div className="text-white text-center h-screen w-screen p-4 flex flex-col items-center ">
      <div>
        <h1 className="text-4xl" style={{ letterSpacing: "2px" }}>M👁️shiVis</h1>
        {/*
          To add more space to the top add padding to the top of the following div
          by changing the pt-4 class to pt-8 or pt-12. (see: https://tailwindcss.com/docs/padding)
   👁️       If you'd like to move this part to the bottom of the screen, change the class to pb-4 or pb-8 and move the following so it is contained by the last one in the page.
          Font size can be changed by changing the text-sm class to text-lg or text-xl. (see : https://tailwindcss.com/docs/font-size)
          As for the links you can use the one below as an example and add more by copying it and changing the href and text.
        */}
        <div className="pt-8 text-sm flex justify-center items-center flex-col mb-0">
          <div className="presentation text-left">
            <p><span className='vis-words'>MoshiVis</span> is an experimental multimodal conversational AI.
              Like <span className='cute-words'>Moshi</span>, MoshiVis can <span className='cute-words'>listen</span> to you and
              <span className='cute-words'> talk</span> at all time for maximum conversational flow.  Now augmented with <span className='vis-words'>visual</span> inputs.</p>
            <p>For instance, you can now ask Moshi to describe your favorite <span className='vis-words'>movie poster</span>,
              grill it on details about the <span className='cute-words'>plot</span>, then go back for more
              details about the <span className='vis-words'>image</span> ask it  to do some <span className='cute-words'>Pirate</span> role play.</p>
            <p>We strive to support all browsers but Chrome works best. Conversations are limited to <span className='cute-words'>5 min</span>.</p>
            <p> Head to the <span className='vis-words'>Settings</span> to configure the image size and other parameters.</p>
            <p> For more information about this project, check out the <a href="https://kyutai.org/2025/03/20/moshivis-blog.html" target="_blank" className="link">MoshiVis project page</a>!</p>
            <p>Baked with &lt;3 @<a href="https://kyutai.org/" className='cute-words underline'>Kyutai</a>.</p>
          </div>
        </div>
      </div>
      <div className="text-sm mt-10 mb-0">
        <p>Add your <span className='cute-words'>email address</span> first, then feel free</p>
        <p> to upload your own <span className='vis-words'>image</span> or select one below.</p>
        <p>Uploaded images should be smaller than <span className='cute-words'>15 MB</span>.</p>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col mb-0">
        {status == 'error' && <p className="text-center text-red-800 text-2xl">{error}</p>}
        {status == 'no_queue' && <p className="text-center">No queue id provided</p>}
        {(status === 'idle' || status === 'bypass') && (
          <>
            {showMicrophoneAccessMessage &&
              <p className="text-center">Please enable your microphone before proceeding</p>
            }
            <Input
              type="email"
              placeholder="Enter your email"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
              error={emailError ?? ""}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  if (modelParams.imageUrl == undefined) {
                    modelParams.setImageUrl("/assets/images/demo/image" + Math.floor(1 + Math.random() * 19) + ".jpg")
                  }
                  onConnect();
                }
              }}
            />
            <Button className="settingsbutton absolute top-4 right-4"
              id={importantSettingsHaveChanged(modelParams) ? 'changed' : 'default'}
              title={importantSettingsHaveChanged(modelParams) ? 'Change or reset MoshiVis here (Your current settings are different from the defaults)' : 'Change or reset settings here'}
              onClick={() => modalRef.current?.showModal()}>Settings</Button>
            <dialog ref={modalRef} className="modal">
              <div className="modal-box border-2 border-white rounded-none flex justify-center bg-black">
                <ModelParams {...modelParams} isConnected={shouldConnect} modal={modalRef} />
              </div>
              <form method="dialog" className="modal-backdrop">
                <button>Close</button>
              </form>
            </dialog>
          </>
        )}
        {status === "connecting" && <p className="text-center">Connecting to queue...</p>}
        {status === "in_queue" && (
          <p className="text-center">
            You're in the queue !<br />
            {currentPosition && <span>Current position: <span className="text-green">{currentPosition}</span></span>}
          </p>)
        }
      </div>
      <div className="mt-0 flex flex-grow justify-center items-center flex-col presentation mb-8">
        <ImageGallery numImages={6} size={110} paramsSetter={modelParams.setImageUrl} clickAction={onConnect}></ImageGallery>
      </div>

      <div className="text-center flex justify-end items-center flex-col">
        <a target="_blank" href="https://kyutai.org/moshi-terms.pdf" className="text-center">Terms of Use</a>
        <a target="_blank" href="https://kyutai.org/moshi-privacy.pdf" className="text-center">Privacy Policy</a>
      </div>
    </div >
  )
};
