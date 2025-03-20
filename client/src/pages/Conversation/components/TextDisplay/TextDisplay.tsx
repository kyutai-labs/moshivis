import { FC, useEffect, useRef } from "react";
import { useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
  displayColor: boolean | undefined;
};

// Palette 2: Purple to Green Moshi
// sns.diverging_palette(288, 145, s=90, l=72, n=11).as_hex()
// Palette 2: Green to orange Moshi
// sns.diverging_palette(145, 40, s=90, l=72, n=11).as_hex()
const textDisplayColors = [
  '#38c886', '#5bd09a', '#80d9af',
  '#a4e2c4', '#c8ead9', '#f2f1f1',
  '#f4e0cb', '#f5cea6', '#f5bd81',
  '#f6ac5b', '#f79b37']

function clamp_color(v: number) {
  return v <= 0
    ? 0
    : v >= textDisplayColors.length
      ? textDisplayColors.length
      : v
}

export const TextDisplay: FC<TextDisplayProps> = ({
  containerRef, displayColor
}) => {
  const { text, textColor } = useServerText();
  const currentIndex = text.length - 1;
  const prevScrollTop = useRef(0);

  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [text]);
  if (displayColor && (textColor.length == text.length)) {
    return (
      <div className="h-full w-full max-w-full max-h-full  p-2 text-white">
        {text.map((t, i) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
            style={{
              color: `${textDisplayColors[clamp_color(textColor[i])]}`
            }}
          >
            {t}
          </span>
        ))
        }
      </div >
    );
  }
  else {
    return (
      <div className="h-full w-full max-w-full max-h-full  p-2 text-white">
        {text.map((t, i) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
          >
            {t}
          </span>
        ))}
      </div>
    );
  };
};
