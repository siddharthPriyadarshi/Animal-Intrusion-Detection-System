import { BASE_URL } from '../constants'

export const getAlertData = async () => {
  try {
    const data = await fetch(BASE_URL)
    // console.log('Data fetched', data)
    return data;
} catch (e) {
    console.log('Error: ', e)
  }
  
  return 0;
}
